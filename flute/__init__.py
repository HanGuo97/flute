import os
import torch
import click
from typing import Callable, cast
from vllm.platforms import current_platform

from . import _C
from . import ops

__version__ = "0.2.4"

QGEMM_SIMPLE_TYPE = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        int,
    ],
    torch.Tensor,
]

QGEMM_RAW_SIMPLE_TYPE = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        int,
        int,
    ],
    None,
]


# we use this instead of `torch.cuda.get_device_capability()` so that
# it works better with multiprocessing (which vLLM uses)
TORCH_CURRENT_DEVICE_CC = current_platform.get_device_capability()

if TORCH_CURRENT_DEVICE_CC == (8, 6):
    click.secho(f"[FLUTE]: Using A6000 with CC={TORCH_CURRENT_DEVICE_CC}", fg="green")
    NUM_SMS = 84

elif TORCH_CURRENT_DEVICE_CC == (8, 0):
    click.secho(f"[FLUTE]: Using A100 with CC={TORCH_CURRENT_DEVICE_CC}", fg="green")
    NUM_SMS = 108

elif TORCH_CURRENT_DEVICE_CC == (8, 9):
    click.secho(f"[FLUTE]: Using RTX4090 with CC={TORCH_CURRENT_DEVICE_CC}", fg="green")
    NUM_SMS = 128

else:
    raise NotImplementedError


QGEMM_SIMPLE_DICT = {
    84 : cast(QGEMM_SIMPLE_TYPE, torch.ops.flute.qgemm_simple_86),
    108: cast(QGEMM_SIMPLE_TYPE, torch.ops.flute.qgemm_simple_80),
    128: cast(QGEMM_SIMPLE_TYPE, torch.ops.flute.qgemm_simple_89),
}

QGEMM_RAW_SIMPLE_DICT = {
    84 : cast(QGEMM_RAW_SIMPLE_TYPE, torch.ops.flute.qgemm_raw_simple_86),
    108: cast(QGEMM_RAW_SIMPLE_TYPE, torch.ops.flute.qgemm_raw_simple_80),
    128: cast(QGEMM_RAW_SIMPLE_TYPE, torch.ops.flute.qgemm_raw_simple_89),
}

qgemm_simple     = QGEMM_SIMPLE_DICT[NUM_SMS]
qgemm_raw_simple = QGEMM_RAW_SIMPLE_DICT[NUM_SMS]


# Load the template configs
if os.environ.get("FLUTE_ABLATIONS", "0") == "1":
    click.secho(f"[FLUTE]: Abalations enabled", fg="yellow")
    TEMPLATE_CONFIGS_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data/qgemm_kernel_raw_generated_configs.ablations.pth")
else:
    TEMPLATE_CONFIGS_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data/qgemm_kernel_raw_generated_configs.pth")

if os.path.exists(TEMPLATE_CONFIGS_PATH):
    TEMPLATE_CONFIGS = torch.load(TEMPLATE_CONFIGS_PATH)
    click.secho(f"[FLUTE]: Template configs loaded from {TEMPLATE_CONFIGS_PATH}", fg="green")
else:
    TEMPLATE_CONFIGS = None
    click.secho(f"[FLUTE]: Template configs not found at {TEMPLATE_CONFIGS_PATH}", fg="red")


# Load the tuned configs
TEMPLATE_TUNED_WITH_M_CONFIGS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data/qgemm_kernel_raw_tuned_configs.pth")
TEMPLATE_TUNED_WITHOUT_M_CONFIGS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data/qgemm_kernel_raw_tuned_configs.no-M.pth")

if os.path.exists(TEMPLATE_TUNED_WITH_M_CONFIGS_PATH):
    TEMPLATE_TUNED_WITH_M_CONFIGS = torch.load(TEMPLATE_TUNED_WITH_M_CONFIGS_PATH)
    click.secho(f"[FLUTE]: Template (tuned, with M) configs loaded from {TEMPLATE_TUNED_WITH_M_CONFIGS_PATH}", fg="green")
else:
    TEMPLATE_TUNED_WITH_M_CONFIGS = None
    click.secho(f"[FLUTE]: Template (tuned, with M) configs not found at {TEMPLATE_TUNED_WITH_M_CONFIGS_PATH}", fg="red")

if os.path.exists(TEMPLATE_TUNED_WITHOUT_M_CONFIGS_PATH):
    TEMPLATE_TUNED_WITHOUT_M_CONFIGS = torch.load(TEMPLATE_TUNED_WITHOUT_M_CONFIGS_PATH)
    click.secho(f"[FLUTE]: Template (tuned, without M) configs loaded from {TEMPLATE_TUNED_WITHOUT_M_CONFIGS_PATH}", fg="green")
else:
    TEMPLATE_TUNED_WITHOUT_M_CONFIGS = None
    click.secho(f"[FLUTE]: Template (tuned, without M) configs not found at {TEMPLATE_TUNED_WITHOUT_M_CONFIGS_PATH}", fg="red")
