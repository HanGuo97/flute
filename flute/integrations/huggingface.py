import os
import enum
import json
import click
import torch
from dataclasses import dataclass
from typing import Tuple, List, Optional

from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.utils import logging, is_accelerate_available
from transformers.quantizers.base import HfQuantizer
from transformers.modeling_utils import PreTrainedModel
from transformers.utils.quantization_config import QuantizationConfigMixin
from transformers.quantizers.auto import (
    AUTO_QUANTIZER_MAPPING,
    AUTO_QUANTIZATION_CONFIG_MAPPING)

from flute.integrations.base import FLUTE_CONFIG_FILE_NAME

logger = logging.get_logger(__name__)


def is_flute_available() -> bool:
    return True


class QuantizationMethod2(str, enum.Enum):
    FLUTE = "flute"


@dataclass
class FluteConfig(QuantizationConfigMixin):

    def __init__(
        self,
        num_bits: int,
        group_size: int,
        num_sms_packed: int,
        example_batch_size: int,
        modules_to_not_convert: Optional[List[str]] = None,
        **kwargs,
    ) -> None:

        if num_bits not in [2, 3, 4]:
            raise ValueError

        self.quant_method = QuantizationMethod2.FLUTE
        self.num_bits = num_bits
        self.group_size = group_size
        self.num_sms_packed = num_sms_packed
        self.example_batch_size = example_batch_size
        self.modules_to_not_convert = modules_to_not_convert

        legacy_template_id_dict_file_name = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../data/qgemm_kernel_raw_tuned_configs.no-M.pth")

        if os.path.exists(legacy_template_id_dict_file_name):
            self.legacy_template_id_dict = torch.load(
                legacy_template_id_dict_file_name,
                weights_only=True)
            click.secho(
                f"[FLUTE]: Template (tuned, without M) configs "
                f"loaded from {legacy_template_id_dict_file_name}",
                fg="green")
        else:
            raise ValueError

    def get_legacy_template_id(
        self,
        N: int,
        K: int,
        dtype: torch.dtype,
    ) -> int:
        return self.legacy_template_id_dict[(
            self.num_sms_packed,
            self.num_bits,
            self.group_size,
            N,
            K,
            str(dtype),
        )]


def _replace_with_flute_linear(
    model: torch.nn.Module,
    quantization_config: FluteConfig,
    modules_to_not_convert: List[str],
    current_key_name: Optional[List[str]] = None,
    has_been_replaced: bool = False,
    pre_quantized: bool = False,
) -> Tuple[torch.nn.Module, bool]:
    from accelerate import init_empty_weights
    from flute.integrations.base import FluteLinear

    if current_key_name is None:
        current_key_name = []

    for name, module in model.named_children():
        current_key_name.append(name)

        if (isinstance(module, torch.nn.Linear)) and name not in modules_to_not_convert:
            # Check if the current key is not in the `modules_to_not_convert`
            current_key_name_str = ".".join(current_key_name)
            if not any(
                (key + "." in current_key_name_str) or (key == current_key_name_str) for key in modules_to_not_convert
            ):
                with init_empty_weights(include_buffers=True):
                    model._modules[name] = FluteLinear(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        num_bits=quantization_config.num_bits,
                        group_size=quantization_config.group_size,
                        template_id=quantization_config.get_legacy_template_id(
                            N=module.out_features,
                            K=module.in_features,
                            dtype=module.weight.dtype),
                        workspace_lazy_init=True,
                        bias=module.bias is not None,
                        device=module.weight.device,
                        dtype=module.weight.dtype)

                    model._modules[name].source_cls = type(module)
                    model._modules[name].requires_grad_(False)
                    model._modules[name].needs_repacking = True
                    has_been_replaced = True

        if len(list(module.children())) > 0:
            _, has_been_replaced = _replace_with_flute_linear(
                module,
                quantization_config=quantization_config,
                modules_to_not_convert=modules_to_not_convert,
                current_key_name=current_key_name,
                has_been_replaced=has_been_replaced,
                pre_quantized=pre_quantized)

        # Remove the last key for recursion
        current_key_name.pop(-1)

    return model, has_been_replaced


def replace_with_flute_linear(
    model: PreTrainedModel,
    quantization_config: FluteConfig,
    modules_to_not_convert: Optional[List[str]] = None,
    current_key_name: Optional[List[str]] = None,
    pre_quantized: bool = False,
) -> PreTrainedModel:

    if modules_to_not_convert is None:
        modules_to_not_convert = ["lm_head"]

    if quantization_config.modules_to_not_convert is not None:
        modules_to_not_convert.extend(quantization_config.modules_to_not_convert)

    modules_to_not_convert = list(set(modules_to_not_convert))
    model, has_been_replaced = _replace_with_flute_linear(
        model=model,  # type: ignore
        quantization_config=quantization_config,
        modules_to_not_convert=modules_to_not_convert,
        current_key_name=current_key_name,
        pre_quantized=pre_quantized)

    if not has_been_replaced:
        logger.warning("You are loading your model using FLUTE quantization "
                       "but no linear modules were found in your model.")

    return model


def _repack_flute_linear(model: torch.nn.Module, quantization_config: FluteConfig) -> None:
    import flute.tune
    import flute.utils
    from flute.integrations.base import FluteLinear

    for name, module in model.named_children():

        if isinstance(module, FluteLinear) and getattr(module, "needs_repacking", False) is True:
            logger.info(f"Repacking {name}...")

            if module.weight.device.type != "cuda":
                device = torch.device("cuda")
                logger.warning(f"Moving {name} to {device} for repacking.")
            else:
                device = module.weight.device

            # reconstruct the unpacked tensor
            Q_unpacked = flute.utils.unpack(
                weight=module.weight.to(device=device),
                scales=module.scales.to(device=device),
                workspace=flute.utils.get_workspace_streamk(device),
                num_bits=module.num_bits,
                group_size=module.group_size,
                template_id_packed=module.template_id,
                num_sms_packed=quantization_config.num_sms_packed)

            # re-pack the tensors
            example_inputs = torch.randn(
                quantization_config.example_batch_size,
                module.in_features,
                dtype=module.scales.dtype,
                device=device)
            Q_repacked, tune_metadata = flute.tune.tune_and_pack(
                inputs=example_inputs,
                weight=Q_unpacked.T.contiguous().to(device="cpu"),
                num_bits=module.num_bits,
                group_size=module.group_size).to(device=module.weight.device)

            if not all([
                not isinstance(module.weight, torch.nn.Parameter),
                module.weight.requires_grad is False,
                Q_repacked.requires_grad is False,
                Q_repacked.shape == module.weight.shape,
                Q_repacked.dtype == module.weight.dtype,
                Q_repacked.device == module.weight.device]):
                raise ValueError

            # sometimes the loading process will cast the `tables2`,
            # hence we might need to re-generate it
            tables2 = flute.utils.make_qmap2_from_qmap(module.tables)
            if not (tables2 == module.tables2).all():
                tables2_casted = tables2.to(dtype=module.tables.dtype).to(dtype=module.tables2.dtype)
                if not (tables2_casted == module.tables2).all():
                    raise ValueError
                logger.warning("The quantization `tables2` are not the same as the "
                               "original ones. Using the newly generated one instead.")

            module.weight = Q_repacked
            module.tables2 = tables2
            module.template_id = tune_metadata.template_id

        if len(list(module.children())) > 0:
            _repack_flute_linear(module, quantization_config=quantization_config)


class FluteHfQuantizer(HfQuantizer):

    requires_calibration = True  # not sure
    required_packages = ["flute-kernel", "accelerate"]

    def __init__(self, quantization_config: FluteConfig, **kwargs) -> None:
        super().__init__(quantization_config, **kwargs)
        self.quantization_config = quantization_config
        self.using_multi_gpu = False
        if not self.pre_quantized:
            raise NotImplementedError

    def validate_environment(self, *args, **kwargs) -> None:
        if not is_flute_available():
            raise ImportError("Loading a FLUTE quantized model requires flute library (`pip install flute-kernel`)")

        if not is_accelerate_available():
            raise ImportError("Loading a FLUTE quantized model requires accelerate library (`pip install accelerate`)")

        device_map = kwargs.get("device_map", None)
        if isinstance(device_map, dict):
            self.using_multi_gpu = len(set(device_map.values())) > 1

    def update_torch_dtype(self, torch_dtype: Optional[torch.dtype]) -> torch.dtype:
        if torch_dtype is None:
            raise TypeError("You did not specify `torch_dtype` in `from_pretrained`.")

        if torch_dtype != torch.float16:
            logger.warning("We suggest you to set `torch_dtype=torch.float16` for better efficiency.")

        return torch_dtype

    def _process_model_before_weight_loading(self, model: PreTrainedModel, keep_in_fp32_modules: Optional[List[str]] = None, **kwargs) -> None:
        from transformers.integrations import get_keys_to_not_convert

        self.modules_to_not_convert = get_keys_to_not_convert(model)
        if not isinstance(self.modules_to_not_convert, list):
            raise TypeError

        if keep_in_fp32_modules is not None:
            # We keep some modules such as the lm_head in their
            # original dtype for numerical stability reasons
            self.modules_to_not_convert.extend(keep_in_fp32_modules)

        if self.quantization_config.modules_to_not_convert is not None:
            # the same code will be duplicated in the next function,
            # but entries will be deduplicated, so it's fine
            self.modules_to_not_convert.extend(self.quantization_config.modules_to_not_convert)

        model = replace_with_flute_linear(
            model=model,
            quantization_config=self.quantization_config,
            modules_to_not_convert=self.modules_to_not_convert,
            pre_quantized=self.pre_quantized)

        model.config.quantization_config = self.quantization_config

    def _process_model_after_weight_loading(self, model: PreTrainedModel, **kwargs) -> None:
        return _repack_flute_linear(model, quantization_config=self.quantization_config)

    def update_missing_keys(self, model, missing_keys: List[str], prefix: str) -> List[str]:
        from flute.integrations.base import FluteLinear

        not_missing_keys = []
        for name, module in model.named_modules():
            if isinstance(module, FluteLinear):
                for missing in missing_keys:
                    if (
                        (name in missing or name in f"{prefix}.{missing}")
                        and missing.endswith(torch.nn.modules.module._EXTRA_STATE_KEY_SUFFIX)
                    ):
                        not_missing_keys.append(missing)
        return [k for k in missing_keys if k not in not_missing_keys]

    @property
    def is_trainable(self) -> bool:
        return False

    @property
    def is_serializable(self) -> bool:
        return True


# Register the quantizer and config
AUTO_QUANTIZER_MAPPING[QuantizationMethod2.FLUTE.value] = FluteHfQuantizer
AUTO_QUANTIZATION_CONFIG_MAPPING[QuantizationMethod2.FLUTE.value] = FluteConfig


def from_pretrained(pretrained_model_name_or_path: str, **kwargs) -> PreTrainedModel:
    config_filename = hf_hub_download(
        repo_id=pretrained_model_name_or_path,
        filename=FLUTE_CONFIG_FILE_NAME,
        revision=kwargs.get("revision", None))

    with open(config_filename) as f:
        flute_config = json.load(f)

    if "num_sms" not in flute_config.keys():
        raise NotImplementedError("Only legacy models are supported for now.")
    else:
        flute_config["num_sms_packed"] = flute_config.pop("num_sms")

    if "example_batch_size" not in kwargs.keys():
        logger.warning(
            "You did not specify `example_batch_size`. Using "
            "a batch size of 1 for the kernel tuning process.")
        flute_config["example_batch_size"] = 1
    else:
        flute_config["example_batch_size"] = kwargs.pop("example_batch_size")

    # load and monkey-patch the model config
    config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        **kwargs)
    config.quantization_config = FluteConfig.from_dict(flute_config)
    return AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        config=config,
        **kwargs)
