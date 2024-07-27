# 2/4
import os
import copy
import torch
import argparse
from collections import defaultdict, Counter
from typing import List, Dict, Tuple


def generate_nested_switch(
    names: List[str],
    cases: Dict[Tuple, str],
    constexprs: List[bool],
    prefix: str,
) -> str:
    INDENT = "    "

    def generate_switch_block(
        depth: int,
        keys: List[Tuple],
    ) -> List[str]:

        indent = INDENT * depth
        code = []
        if depth < len(names):
            # header
            var_name = names[depth]
            constexpr = constexprs[depth]
            if constexpr is False:
                code.append(f"{indent}switch ({var_name})")
                code.append(f"{indent}{{")

            # get the unique cases for this variable
            cases_for_var = set()
            for key in keys:
                if len(key) <= depth:
                    raise ValueError(f"Key {key} is too short for depth {depth}")
                cases_for_var.add(key[depth])

            for case_index, case in enumerate(sorted(cases_for_var)):
                if constexpr is False:
                    code.append(f"{indent}case {case}:")
                else:
                    if isinstance(case, int):
                        case_string = f"cute::Int<{case}>"
                    else:
                        case_string = case

                    if case_index == 0:
                        code.append(f"{indent}if constexpr (cute::is_same_v<{var_name}, {case_string}>)")
                    else:
                        code.append(f"{indent}else if constexpr (cute::is_same_v<{var_name}, {case_string}>)")
                    code.append(f"{indent}{{")

                # filter the keys that match this case
                filtered_keys = [key for key in keys if key[depth] == case]

                if depth < len(names) - 1:
                    code.extend(generate_switch_block(depth + 1, filtered_keys))
                else:
                    if len(filtered_keys) != 1:
                        raise ValueError(f"This cannot happen: {filtered_keys}")
                    command = cases[filtered_keys[0]]
                    code.append(f"{indent}{INDENT}{command}")
                if constexpr is False:
                    code.append(f"{indent}{INDENT}break;")
                else:
                    code.append(f"{indent}}}")

            # footer
            if constexpr is False:
                code.append(f"{indent}default:")
                code.append(f"{indent}{INDENT}AT_ERROR(\"Unsupported {var_name} value\");")
                code.append(f"{indent}}}")
            else:
                code.append(f"{indent}else")
                code.append(f"{indent}{{")
                code.append(f"{indent}{INDENT}AT_ERROR(\"Unsupported {var_name} value\");")
                code.append(f"{indent}}}")

        return code

    all_keys = list(cases.keys())
    code = generate_switch_block(0, all_keys)
    code = [f"{prefix}{_code}" for _code in code]
    return "\n".join(code)


def codegen_tuned(no_M_specialization: bool = False) -> None:
    file_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(file_dir, "csrc/qgemm_kernel_generated.cu")
    cfg_path = os.path.join(file_dir, "data/qgemm_kernel_raw_generated_configs.pth")
    tuned_path = os.path.join(file_dir, "data/qgemm_kernel_raw_tuned_configs.pth")

    configs = torch.load(cfg_path)
    templates_with_M = torch.load(tuned_path)
    templates_without_M = {}

    # finding the most common template for each key without M
    KEY_M_INDEX = 3
    templates_without_M_raw = defaultdict(list)
    for keys, template_id in templates_with_M.items():
        keys_no_M = keys[:(KEY_M_INDEX)] + keys[(KEY_M_INDEX + 1):]
        templates_without_M_raw[keys_no_M].append(template_id)
    for keys_no_M, template_ids in templates_without_M_raw.items():
        counter = Counter(template_ids)
        most_common_template_id, _ = counter.most_common(1)[0]
        templates_without_M[keys_no_M] = most_common_template_id

    if no_M_specialization is False:
        templates = templates_with_M
    else:
        templates = templates_without_M

    cases = {}
    for keys, template_id in templates.items():
        NumBits = keys[1]
        config = configs[(NumBits, template_id)]

        # https://www.w3schools.com/python/gloss_python_change_tuple_item.asp
        new_keys = list(copy.deepcopy(keys))
        if new_keys[-1] == "torch.float16":
            new_keys[-1] = "cute::half_t"
        elif new_keys[-1] == "torch.bfloat16":
            new_keys[-1] = "cute::bfloat16_t"
        else:
            raise ValueError(f"Unsupported dtype: {new_keys[-1]}")

        cases[tuple(new_keys)] = (
            f'RUN_QGEMM('
            f'T, TQ, T2, '
            f'{config["Slices"]}, '
            f'{config["SMs"]} * SMs::value, '
            f'{config["Threads"]}, '
            f'{config["TileM"]}, '
            f'{config["TileK"]}, '
            f'{config["TileP"]}, '
            f'{config["Stages"]}, '
            f'NumBits::value, '
            f'GroupSize::value, '
            f'{config["QuantMapMode"]}, '
            f'{config["AccumulationMode"]}, '
            f'{config["DecompositionMode"]}, '
            f'{config["G2STiledCopySizeS"]}, '
            f'{config["MmaPrmK"]});'
        )

    if no_M_specialization is False:
        names      = ["SMs", "NumBits", "GroupSize", "M"  , "N"  , "K"  , "T"]
        constexprs = [True , True     , True       , False, False, False, True]
    else:
        names      = ["SMs", "NumBits", "GroupSize"       , "N"  , "K"  , "T"]
        constexprs = [True , True     , True              , False, False, True]

    code_append = generate_nested_switch(
        names=names,
        cases=cases,
        constexprs=constexprs,
        prefix="    ")

    code = []
    with open(out_path) as f:
        for line in f.readlines():
            code.append(line)
            if line.startswith("    // Generated Code Below"):
                code.append(code_append)
                code.append("\n")

    save_path = os.path.join(file_dir, "data/qgemm_kernel_raw_tuned_configs.no-M.pth")
    torch.save(templates_without_M, save_path)
    with open(out_path, "w") as f:
        f.write("".join(code))


def codegen_raw() -> None:
    options_Slices = [0]
    options_SMs = [1, 2, 4]
    options_Tiles = [
        (256, 32, 64, 64),
        (256, 32, 64, 32),
        (128, 16, 64, 32),
    ]
    options_Stages = [2, 3, 4, 5]
    options_QuantMapMode = [
        "kVectorized   ",
        "kVectorized_32",
        "kVectorized_16",
        "kVectorized_8 ",
    ]
    options_AccumulationMode = ["kMixed"]
    options_DecompositionMode = ["kStreamK"]
    options_G2STiledCopySizeS = [2]
    options_MmaPrmK = [1]

    cases = {}
    configs = {}
    for NumBits in [4, 3, 2]:
        index = 0
        for Slices in options_Slices:
            for SMs in options_SMs:
                for (Threads, TileM, TileK, TileP) in options_Tiles:
                    for Stages in options_Stages:
                        for QuantMapMode in options_QuantMapMode:
                            if NumBits != 4 and QuantMapMode != options_QuantMapMode[0]:
                                continue
                            for AccumulationMode in options_AccumulationMode:
                                for DecompositionMode in options_DecompositionMode:
                                    for G2STiledCopySizeS in options_G2STiledCopySizeS:
                                        for MmaPrmK in options_MmaPrmK:
                                            cases[(NumBits, index)] = (
                                                f'RUN_QGEMM('
                                                f'T, TQ, T2, '
                                                f'{Slices}, '
                                                f'{SMs} * SMs::value, '
                                                f'{Threads}, '
                                                f'{TileM}, '
                                                f'{TileK}, '
                                                f'{TileP}, '
                                                f'{Stages}, '
                                                f'NumBits::value, '
                                                f'GroupSize::value, '
                                                f'{QuantMapMode}, '
                                                f'{AccumulationMode}, '
                                                f'{DecompositionMode}, '
                                                f'{G2STiledCopySizeS}, '
                                                f'{MmaPrmK});'
                                            )
                                            configs[(NumBits, index)] = {
                                                "Slices": Slices,
                                                "SMs": SMs,
                                                "Threads": Threads,
                                                "TileM": TileM,
                                                "TileK": TileK,
                                                "TileP": TileP,
                                                "Stages": Stages,
                                                "QuantMapMode": QuantMapMode,
                                                "AccumulationMode": AccumulationMode,
                                                "DecompositionMode": DecompositionMode,
                                                "G2STiledCopySizeS": G2STiledCopySizeS,
                                                "MmaPrmK": MmaPrmK,
                                            }
                                            index = index + 1

    names      = ["NumBits", "template_id"]
    constexprs = [True     , False]
    code_append = generate_nested_switch(
        names=names,
        cases=cases,
        constexprs=constexprs,
        prefix="    ")

    file_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(file_dir, "csrc/qgemm_kernel_raw_generated.cu")
    cfg_path = os.path.join(file_dir, "data/qgemm_kernel_raw_generated_configs.pth")

    code = []
    with open(out_path) as f:
        for line in f.readlines():
            code.append(line)
            if line.startswith("    // Generated Code Below"):
                code.append(code_append)
                code.append("\n")

    torch.save(configs, cfg_path)
    with open(out_path, "w") as f:
        f.write("".join(code))


def codegen_ablations() -> None:
    options_Slices = [0]
    options_SMs = [1, 2, 4]
    options_Tiles = [
        (256, 32, 64, 64),
        (256, 32, 64, 32),
        (128, 16, 64, 32),
    ]
    options_Stages = [2, 3, 4, 5]
    options_QuantMapMode = [
        "kVectorized   ",
        "kVectorized_32",
        "kVectorized_16",
        "kVectorized_8 ",
        "config::QuantMapModeEnum::Basic",
    ]
    options_AccumulationMode = [
        "kMixed",
        "config::AccumulationModeEnum::Low",
        "config::AccumulationModeEnum::High",
    ]
    options_DecompositionMode = ["kStreamK"]
    options_G2STiledCopySizeS = [2]
    options_MmaPrmK = [1]

    cases = {}
    configs = {}
    for NumBits in [4]:
        index = 0
        for Slices in options_Slices:
            for SMs in options_SMs:
                for (Threads, TileM, TileK, TileP) in options_Tiles:
                    for Stages in options_Stages:
                        for QuantMapMode in options_QuantMapMode:
                            if NumBits != 4 and QuantMapMode != options_QuantMapMode[0]:
                                continue
                            for AccumulationMode in options_AccumulationMode:

                                # we want to ablate one of these
                                ablating_QuantMapMode = QuantMapMode.startswith("config::")
                                ablating_AccumulationMode = AccumulationMode.startswith("config::")
                                # https://stackoverflow.com/questions/432842/how-do-you-get-the-logical-xor-of-two-variables-in-python
                                if not (bool(ablating_QuantMapMode) != bool(ablating_AccumulationMode)):
                                    continue

                                for DecompositionMode in options_DecompositionMode:
                                    for G2STiledCopySizeS in options_G2STiledCopySizeS:
                                        for MmaPrmK in options_MmaPrmK:
                                            cases[(NumBits, index)] = (
                                                f'RUN_QGEMM('
                                                f'T, TQ, T2, '
                                                f'{Slices}, '
                                                f'{SMs} * SMs::value, '
                                                f'{Threads}, '
                                                f'{TileM}, '
                                                f'{TileK}, '
                                                f'{TileP}, '
                                                f'{Stages}, '
                                                f'NumBits::value, '
                                                f'GroupSize::value, '
                                                f'{QuantMapMode}, '
                                                f'{AccumulationMode}, '
                                                f'{DecompositionMode}, '
                                                f'{G2STiledCopySizeS}, '
                                                f'{MmaPrmK});'
                                            )
                                            configs[(NumBits, index)] = {
                                                "Slices": Slices,
                                                "SMs": SMs,
                                                "Threads": Threads,
                                                "TileM": TileM,
                                                "TileK": TileK,
                                                "TileP": TileP,
                                                "Stages": Stages,
                                                "QuantMapMode": QuantMapMode,
                                                "AccumulationMode": AccumulationMode,
                                                "DecompositionMode": DecompositionMode,
                                                "G2STiledCopySizeS": G2STiledCopySizeS,
                                                "MmaPrmK": MmaPrmK,
                                            }
                                            index = index + 1

    names      = ["NumBits", "template_id"]
    constexprs = [True     , False]
    code_append = generate_nested_switch(
        names=names,
        cases=cases,
        constexprs=constexprs,
        prefix="    ")

    file_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(file_dir, "csrc/qgemm_kernel_raw_generated.cu")
    cfg_path = os.path.join(file_dir, "data/qgemm_kernel_raw_generated_configs.ablations.pth")

    code = []
    with open(out_path) as f:
        for line in f.readlines():
            code.append(line)
            if line.startswith("    // Generated Code Below"):
                code.append(code_append)
                code.append("\n")

    torch.save(configs, cfg_path)
    with open(out_path, "w") as f:
        f.write("".join(code))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuned", action="store_true")
    parser.add_argument("--tuned-no-M", action="store_true")
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--ablations", action="store_true")
    args = parser.parse_args()

    if args.tuned is True:
        codegen_tuned()
    if args.tuned_no_M is True:
        codegen_tuned(no_M_specialization=True)
    if args.raw is True:
        codegen_raw()
    if args.ablations is True:
        codegen_ablations()

