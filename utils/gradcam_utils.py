# utils/gradcam_utils.py
import numpy as np
from PIL import Image
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from model.deepfake_model import load_model as load_timm_model

# Default image size (match training)
IMG_SIZE = 224

# Simple albumentations transform used for inference
_preprocess = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def _find_target_layer(model):
    """
    Try to find a reasonable conv layer for Grad-CAM.
    - Prefer attribute 'conv_head' (common in timm EfficientNet)
    - Otherwise pick last Conv2d module in model.modules()
    """
    if hasattr(model, "conv_head"):
        return [model.conv_head]
    # search reversed modules for first Conv2d
    import torch.nn as nn
    modules = list(model.modules())[::-1]
    for m in modules:
        if isinstance(m, nn.Conv2d):
            return [m]
    # fallback to model itself (may fail)
    return [model]

def load_model(path: str = "model/deepfake_model.pth", device: str = "cpu"):
    """
    Convenience wrapper that calls the model loader from model/deepfake_model.py
    """
    return load_timm_model(path=path, device=device)

def _preprocess_pil(pil: Image.Image):
    """
    Returns a tensor (1, C, H, W) on CPU (caller may move to device).
    Also returns original resized RGB numpy image scaled [0,1] for show_cam_on_image
    """
    orig = pil.convert("RGB")
    # resize for heatmap visualization
    vis = orig.resize((IMG_SIZE, IMG_SIZE))
    vis_np = np.array(vis).astype(np.float32) / 255.0  # H,W,3 in [0,1]
    img_np = np.array(orig).astype(np.float32) / 255.0
    tensor = _preprocess(image=img_np)["image"].unsqueeze(0)  # torch tensor, float32
    return tensor, vis_np

def predict_and_heatmap(model, image, device: str = "cpu", img_size: int = IMG_SIZE) -> dict:
    """
    model: loaded torch model (already on device or will be moved)
    image: PIL.Image.Image instance (or path string)
    Returns dict: {'verdict': str, 'confidence': float, 'heatmap': np.ndarray (H,W,3) RGB uint8}
    """
    # accept path or PIL
    if isinstance(image, str):
        pil = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        pil = image
    else:
        raise ValueError("image must be PIL.Image or path string")

    tensor, vis_rgb = _preprocess_pil(pil)  # tensor on cpu
    tensor = tensor.to(device)

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(tensor)  # shape [1, num_classes]
        probs = torch.softmax(outputs, dim=1)[0]  # tensor
        # Detach and move to cpu safely
        probs_np = probs.detach().cpu().numpy()
        cls = int(probs_np.argmax())
        confidence = float(probs_np[cls])

    # Grad-CAM
    try:
        target_layers = _find_target_layer(model)
        use_cuda = device.startswith("cuda")
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)
        grayscale_cam = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(cls)])[0]  # HxW float [0,1]
        # show_cam_on_image expects float [0,1]
        cam_image = show_cam_on_image(vis_rgb, grayscale_cam, use_rgb=True)
        # cam_image is uint8 RGB by default from utility
        heatmap = cam_image
    except Exception as e:
        # If CAM fails, return None for heatmap but still provide verdict
        heatmap = None

    verdict = "FAKE" if cls == 1 else "REAL"
    return {"verdict": verdict, "confidence": confidence, "heatmap": heatmap}
