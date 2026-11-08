"""Model factory: student, Mamba teachers and size-matched baselines.

Teachers / baselines and where they come from
  mambavision_t / _t2 / _s / _b : `pip install mambavision`  (NVIDIA, CVPR 2025)
  vmamba_t                       : official repo github.com/MzeroMiko/VMamba (set VMAMBA_REPO)
  convnextv2_t, swin_t, efficientvit_b1, mobilenetv4_m, mobilenetv3_s, resnet50 : timm
"""
from __future__ import annotations

import os
import sys

import torch.nn as nn

from .micropathmamba import micropathmamba

TIMM = {
    "convnextv2_t": "convnextv2_tiny.fcmae_ft_in1k",
    "swin_t": "swin_tiny_patch4_window7_224.ms_in1k",
    "efficientvit_b1": "efficientvit_b1.r224_in1k",
    "mobilenetv4_m": "mobilenetv4_conv_medium.e500_r224_in1k",
    "mobilenetv3_s": "mobilenetv3_small_100.lamb_in1k",
    "resnet50": "resnet50.a1_in1k",
    "deit_t": "deit_tiny_patch16_224.fb_in1k",
}
MAMBAVISION = {
    "mambavision_t": "mambavision_tiny_1k",
    "mambavision_t2": "mambavision_tiny2_1k",
    "mambavision_s": "mambavision_small_1k",
    "mambavision_b": "mambavision_base_1k",
}
STUDENTS = {
    "micropathmamba": dict(backbone="mnv2"),                    # ImageNet-pretrained MobileNetV2 + 2 SSD-Mamba blocks
    "micropathmamba_cnn": dict(backbone="mnv2", depth_ssm=0),   # ablation: same student without SSM blocks
    "micropathmamba_tiny": dict(backbone="custom"),             # ablation: 0.95 M from-scratch backbone
}


def _replace_head(m: nn.Module, num_classes: int) -> nn.Module:
    if hasattr(m, "reset_classifier"):
        m.reset_classifier(num_classes)
        return m
    for attr in ("head", "fc", "classifier"):
        h = getattr(m, attr, None)
        if isinstance(h, nn.Linear):
            setattr(m, attr, nn.Linear(h.in_features, num_classes))
            return m
    raise RuntimeError("could not find classifier head to replace")


def build_model(name: str, num_classes: int = 9, pretrained: bool = True) -> nn.Module:
    if name in STUDENTS:
        return micropathmamba(num_classes=num_classes, pretrained=pretrained, **STUDENTS[name])
    if name in TIMM:
        import timm
        return timm.create_model(TIMM[name], pretrained=pretrained, num_classes=num_classes)
    if name in MAMBAVISION:
        from mambavision import create_model     # pip install mambavision (needs mamba-ssm, Linux + CUDA)
        m = create_model(MAMBAVISION[name], pretrained=pretrained, model_path=f"/tmp/{name}.pth.tar")
        return _replace_head(m, num_classes)
    if name == "vmamba_t":
        # VMamba has no pip package. Clone the repo, build its selective-scan kernel, download
        # the VMamba-T checkpoint, then point VMAMBA_REPO / VMAMBA_CFG / VMAMBA_CKPT at them.
        repo = os.environ["VMAMBA_REPO"]
        sys.path.insert(0, repo)
        from classification.config import get_config   # noqa: E402  (repo module)
        from classification.models import build_model as vm_build   # noqa: E402
        import argparse
        import torch
        cfg = get_config(argparse.Namespace(cfg=os.environ["VMAMBA_CFG"], opts=None))
        m = vm_build(cfg)
        if pretrained and os.environ.get("VMAMBA_CKPT"):
            sd = torch.load(os.environ["VMAMBA_CKPT"], map_location="cpu")
            m.load_state_dict(sd.get("model", sd), strict=False)
        return _replace_head(m, num_classes)
    raise KeyError(f"unknown model {name}. Options: {list(STUDENTS) + list(TIMM) + list(MAMBAVISION) + ['vmamba_t']}")
