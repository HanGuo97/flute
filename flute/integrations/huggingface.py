import enum
import json
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
        modules_to_not_convert: Optional[List[str]] = None,
        **kwargs,
    ) -> None:

        if num_bits not in [2, 3, 4]:
            raise ValueError

        self.quant_method = QuantizationMethod2.FLUTE
        self.num_bits = num_bits
        self.group_size = group_size
        self.num_sms_packed = num_sms_packed
        self.modules_to_not_convert = modules_to_not_convert


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
                num_sms_packed=quantization_config.num_sms_packed)

            # re-pack the tensors
            Q_repacked = flute.utils.pack(
                Q_unpacked.T.contiguous().to(device="cpu"),
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
            module.weight = Q_repacked

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
            logger.info("You did not specify `torch_dtype` in `from_pretrained`. Setting it to `torch.float16`.")
            torch_dtype = torch.float16

        if torch_dtype != torch.float16:
            logger.warning("We suggest you to set `torch_dtype=torch.float16` for better efficiency with AWQ.")

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
        flute_config["num_sms_packed"] = flute_config.pop("num_sms")

    # load and monkey-patch the model config
    config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        **kwargs)
    config.quantization_config = FluteConfig.from_dict(flute_config)
    return AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        config=config,
        **kwargs)
