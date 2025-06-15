import torch
from torch import nn

from lerobot.common.lora.loralinear import LoRaLinear
from lerobot.configs.lora import LoRaConfig

import gc


def trigger_gc():
    # make sure all kernels are finished
    # which allow the memory to be freed
    torch.cuda.synchronize()

    # trigger the destructor of the tensors
    gc.collect()

    # reclaim cuda memory
    torch.cuda.empty_cache()


def _recursive_setattr(model: nn.Module, module_name: str, new_module: nn.Module):
    split_list = module_name.split(".")
    current_module = model
    for name in split_list[:-1]:
        current_module = getattr(current_module, name)
    current_module.__setattr__(split_list[-1], new_module)


_LAYER_REPLACEMENTS = {
    nn.Linear: LoRaLinear,
}

_LAYER_INV_REPLACEMENTS = {
    LoRaLinear: nn.Linear,
}


def _no_further_replacement(module: nn.Module) -> bool:
    if hasattr(module, "_no_further_replacement"):
        return module._no_further_replacement


def replace_layers(
    module: nn.Module,
    lora_config: LoRaConfig,
    device="cuda",
    name_prefix="",
) -> nn.Module:
    module_type = type(module)

    replacements = _LAYER_REPLACEMENTS

    print(f"Replace {name_prefix} ({module_type})?")

    if (module_type in replacements) and not _no_further_replacement(module):
        new_module_type = replacements[module_type]

        print(f"Replacing {name_prefix} ({module_type} to {new_module_type}) ...")

        new_module = new_module_type.replace(
            module, lora_config=lora_config, device=device
        )

        # delete the previous module to save memory
        if new_module != module:
            del module

        # trigger GC to save memory
        trigger_gc()

        # point to the new module so that we recurse on it
        module = new_module

    # only replace the immediate children and not all children
    # as we might update the structure quite a lot through our replacements
    # and original children might not have counterparts anymore after replacements
    # (e.g. q, k, v projs will not exist anymore in the attention after replacement to our attention classes)
    if not _no_further_replacement(module):
        for sub_module_name, sub_module in module.named_children():
            full_module_name = (
                name_prefix + "." + sub_module_name
                if name_prefix != ""
                else sub_module_name
            )

            # replace modules in this module (updated or not) recursively
            new_sub_module = replace_layers(
                sub_module,
                lora_config=lora_config,
                device=device,
                name_prefix=full_module_name,
            )

            _recursive_setattr(module, sub_module_name, new_sub_module)

    return module


def replace_back_layers(
    module: nn.Module,
    lora_config: LoRaConfig,
    device="cuda",
    name_prefix="",
) -> nn.Module:
    module_type = type(module)

    replacements = _LAYER_INV_REPLACEMENTS

    print(f"Replace back {name_prefix} ({module_type})?")

    if module_type in replacements:
        new_module_type = replacements[module_type]

        print(f"Replacing back {name_prefix} ({module_type} to {new_module_type}) ...")

        new_module = module.replace_back()

        # delete the previous module to save memory
        if new_module != module:
            del module

        # point to the new module so that we recurse on it
        module = new_module

    # only replace the immediate children and not all children
    # as we might update the structure quite a lot through our replacements
    # and original children might not have counterparts anymore after replacements
    # (e.g. q, k, v projs will not exist anymore in the attention after replacement to our attention classes)
    for sub_module_name, sub_module in module.named_children():
        full_module_name = (
            name_prefix + "." + sub_module_name
            if name_prefix != ""
            else sub_module_name
        )

        # replace modules in this module (updated or not) recursively
        new_sub_module = replace_layers(
            sub_module,
            lora_config=lora_config,
            device=device,
            name_prefix=full_module_name,
        )

        _recursive_setattr(module, sub_module_name, new_sub_module)

    return module
