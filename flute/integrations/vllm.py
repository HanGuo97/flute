import os
import sys
import runpy
import click
from copy import deepcopy
from . import vllm_utils


def patch_vllm() -> None:
    from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
    # click.secho(f"vLLM supported quantization methods: {QUANTIZATION_METHODS.keys()}", fg="green")
    if "flute" in QUANTIZATION_METHODS.keys():
        raise ValueError("flute quantization method is already supported in vLLM")
    QUANTIZATION_METHODS["flute"] = vllm_utils.FluteConfig
    # click.secho(f"vLLM supports quantization methods: {QUANTIZATION_METHODS.keys()}", fg="green")


def main() -> None:
    # adding flute to vLLM
    patch_vllm()

    if len(sys.argv) > 1:
        # Get the module or script name/path
        target = sys.argv[1]

        # Capture the remaining arguments
        sys.argv = deepcopy(sys.argv[1:])

        if os.path.isfile(target):
            # Run as a script
            runpy.run_path(target, run_name="__main__")
        else:
            # Run as a module
            runpy.run_module(target, run_name="__main__", alter_sys=True)
    else:
        print("No command or script provided to execute in vLLM")


if __name__ == "__main__":
    main()