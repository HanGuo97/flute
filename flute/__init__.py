import os
import torch
import click
from typing import Callable, cast
from . import _C
from . import ops

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


TORCH_CURRENT_DEVICE_CC = torch.cuda.get_device_capability()

if TORCH_CURRENT_DEVICE_CC == (8, 6):
    click.secho(f"[FLUTE]: Using A6000 with CC={TORCH_CURRENT_DEVICE_CC}", fg="green")
    NUM_SMS          = 84
    qgemm_simple     = cast(QGEMM_SIMPLE_TYPE    , torch.ops.flute.qgemm_simple_86)
    qgemm_raw_simple = cast(QGEMM_RAW_SIMPLE_TYPE, torch.ops.flute.qgemm_raw_simple_86)

elif TORCH_CURRENT_DEVICE_CC == (8, 0):
    click.secho(f"[FLUTE]: Using A100 with CC={TORCH_CURRENT_DEVICE_CC}", fg="green")
    NUM_SMS          = 108
    qgemm_simple     = cast(QGEMM_SIMPLE_TYPE    , torch.ops.flute.qgemm_simple_80)
    qgemm_raw_simple = cast(QGEMM_RAW_SIMPLE_TYPE, torch.ops.flute.qgemm_raw_simple_80)

else:
    raise NotImplementedError


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
