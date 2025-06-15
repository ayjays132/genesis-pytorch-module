"""Utilities for attaching GENESIS plugins to existing PyTorch models."""

import torch
import torch.nn as nn

from . import GenesisPlugin


def attach_genesis_plugin(base_model: nn.Module, plugin: GenesisPlugin, layer_name: str,
                           output_attr: str = "genesis_logits", with_grad: bool = False):
    """Attach a GenesisPlugin to a given layer of a base model using a forward hook.

    Parameters
    ----------
    base_model : nn.Module
        The model to augment. The specified layer must exist within this model.
    plugin : GenesisPlugin
        The plugin instance providing GENESIS functionality.
    layer_name : str
        Name of the layer within ``base_model`` whose output will be fed to
        ``plugin``.
    output_attr : str, optional
        Attribute name on ``base_model`` where the plugin's logits will be stored
        after each forward pass. Defaults to ``"genesis_logits"``.
    with_grad : bool, optional
        If ``True``, execute the plugin with gradient tracking so that gradients
        can propagate from any loss computed using ``output_attr`` back into the
        hooked layer. Defaults to ``False`` which disables gradient tracking.

    Returns
    -------
    torch.utils.hooks.RemovableHandle
        The handle of the registered forward hook. Store it if you wish to
        remove the hook later.
    """
    modules = dict(base_model.named_modules())
    if layer_name not in modules:
        raise ValueError(f"Layer '{layer_name}' not found in model")
    layer = modules[layer_name]

    # Determine the device of the chosen layer by inspecting its parameters or
    # buffers. If no parameters are present (e.g. an activation layer), fall
    # back to the device of the base model or CPU.
    device = None
    for param in layer.parameters(recurse=False):
        device = param.device
        break
    if device is None:
        for buf in layer.buffers(recurse=False):
            device = buf.device
            break
    if device is None:
        try:
            device = next(base_model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    # Move the plugin to the same device to avoid mismatched tensor devices
    plugin.to(device)

    def hook(module, inp, out):
        if with_grad:
            logits, _, _ = plugin(out)
        else:
            with torch.no_grad():
                logits, _, _ = plugin(out)
        setattr(base_model, output_attr, logits)
        return out

    return layer.register_forward_hook(hook)
