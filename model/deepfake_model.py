# model/deepfake_model.py
import os
import torch
import timm
from collections import OrderedDict

def build_model(num_classes: int = 2, backbone: str = "efficientnet_b0", pretrained: bool = False):
    """
    Build a timm model instance (not loaded with weights).
    """
    model = timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)
    return model

def load_model(path: str = "model/deepfake_model.pth", device: str = "cpu", backbone: str = "efficientnet_b0"):
    """
    Load a model state dict from disk safely, map to device, handle DataParallel keys.
    Returns a torch.nn.Module on the requested device.
    """
    model = build_model(num_classes=2, backbone=backbone, pretrained=False)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    # load state dict with map_location
    state = torch.load(path, map_location=device)

    # some checkpoints save whole model; handle both cases
    if isinstance(state, dict) and "state_dict" in state:
        state_dict = state["state_dict"]
    else:
        state_dict = state

    # handle 'module.' prefix from DataParallel
    new_state = OrderedDict()
    for k, v in state_dict.items():
        new_k = k
        if k.startswith("module."):
            new_k = k[len("module."):]
        new_state[new_k] = v

    model.load_state_dict(new_state)
    model.to(device)
    model.eval()
    return model
