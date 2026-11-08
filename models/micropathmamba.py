"""
MicroPathMamba: an NPU-friendly vision Mamba student for the STM32N6 Neural-ART NPU.

Design rules (so ST Edge AI can map the graph statically and INT8 works):
  * fixed input size, fixed token count (7 x 7 = 49 at the last stage)
  * convolutional stages 1-4 (inverted residual, BatchNorm, ReLU6) -> run on the NPU
  * stage 5: MambaVision-style mixer whose selective scan is written in matrix
    (state-space-duality, Mamba-2) form: MatMul / Mul / Sub / Abs / Exp / Sigmoid only.
    No custom CUDA kernels, no Loop / Scan / CumSum ONNX ops.
  * BatchNorm instead of LayerNorm (folds into convs at export)
  * dt = dt_max * sigmoid(.) instead of softplus (bounded, quantization friendly)

The matrix form is proven equal to a forward + backward selective scan in
tests/test_ssd_equivalence.py.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_act(cin, cout, k=3, s=1, groups=1, act=True):
    layers = [nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False), nn.BatchNorm2d(cout)]
    if act:
        layers.append(nn.ReLU6(inplace=True))
    return nn.Sequential(*layers)


class InvertedResidual(nn.Module):
    def __init__(self, cin, cout, stride, expand):
        super().__init__()
        hid = cin * expand
        self.use_res = stride == 1 and cin == cout
        self.block = nn.Sequential(
            conv_bn_act(cin, hid, 1) if expand != 1 else nn.Identity(),
            conv_bn_act(hid, hid, 3, stride, groups=hid),
            conv_bn_act(hid, cout, 1, act=False),
        )

    def forward(self, x):
        y = self.block(x)
        return x + y if self.use_res else y


class SSDMixer(nn.Module):
    """Bidirectional selective state-space mixer in matrix (SSD) form.

    x: (B, C, H, W) -> (B, C, H, W). Sequence length L = H*W must equal `tokens`.
    Half of the inner channels go through the SSM, the other half through a
    symmetric depthwise-conv branch (as in MambaVision), then they are concatenated.
    """

    def __init__(self, dim, tokens=49, d_state=16, heads=4, expand=2, dt_max=0.5):
        super().__init__()
        inner = dim * expand
        assert inner % 2 == 0 and (inner // 2) % heads == 0
        self.half = inner // 2
        self.heads, self.d_state, self.tokens, self.dt_max = heads, d_state, tokens, dt_max
        self.head_dim = self.half // heads

        self.in_proj = nn.Conv2d(dim, inner, 1, bias=False)
        self.dw_ssm = conv_bn_act(self.half, self.half, 3, groups=self.half, act=False)
        self.dw_sym = conv_bn_act(self.half, self.half, 3, groups=self.half, act=False)
        # per-token SSM parameters: B, C (shared across heads) and dt (per head)
        self.bcdt = nn.Conv2d(self.half, 2 * d_state + heads, 1, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.linspace(1.0, 16.0, heads)))   # A = -exp(A_log)
        self.D = nn.Parameter(torch.ones(heads))
        self.out_proj = nn.Conv2d(inner, dim, 1, bias=False)
        self.out_bn = nn.BatchNorm2d(dim)
        # constant upper-triangular ones: cumsum as a MatMul (no CumSum op in the graph)
        self.register_buffer("tri", torch.triu(torch.ones(tokens, tokens)), persistent=False)

    def forward(self, x):
        b, _, h, w = x.shape
        L = h * w
        assert L == self.tokens, f"expected {self.tokens} tokens, got {L}"
        u = self.in_proj(x)
        xs, zs = u[:, : self.half], u[:, self.half:]
        xs = F.silu(self.dw_ssm(xs))
        zs = F.silu(self.dw_sym(zs))

        p = self.bcdt(xs).flatten(2)                                   # (b, 2N+H, L)
        Bm = p[:, : self.d_state].transpose(1, 2)                      # (b, L, N)
        Cm = p[:, self.d_state: 2 * self.d_state].transpose(1, 2)      # (b, L, N)
        dt = self.dt_max * torch.sigmoid(p[:, 2 * self.d_state:])      # (b, H, L)

        A = -torch.exp(self.A_log).view(1, -1, 1)                      # (1, H, 1)
        c = torch.matmul(dt * A, self.tri)                             # cumsum over tokens (b, H, L)
        decay = torch.exp(-torch.abs(c.unsqueeze(-1) - c.unsqueeze(-2)))   # (b, H, L, L)
        scores = torch.matmul(Cm, Bm.transpose(1, 2)).unsqueeze(1)     # (b, 1, L, L)

        xv = xs.flatten(2).view(b, self.heads, self.head_dim, L).transpose(2, 3)   # (b, H, L, P)
        y = torch.matmul(decay * scores, dt.unsqueeze(-1) * xv)         # (b, H, L, P)
        y = y + self.D.view(1, -1, 1, 1) * xv
        y = y.transpose(2, 3).reshape(b, self.half, h, w)

        out = torch.cat([y, zs], dim=1)
        return self.out_bn(self.out_proj(out))


class MambaBlock(nn.Module):
    def __init__(self, dim, tokens, d_state, heads, mlp_ratio=2.0):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.mixer = SSDMixer(dim, tokens, d_state, heads)
        self.norm2 = nn.BatchNorm2d(dim)
        hid = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Conv2d(dim, hid, 1), nn.ReLU6(inplace=True), nn.Conv2d(hid, dim, 1))

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class MicroPathMamba(nn.Module):
    """Hybrid CNN + SSD-Mamba classifier for 224x224 H&E patches.

    backbone="mnv2"   : ImageNet-pretrained MobileNetV2-1.0 features (timm, stride 32, 320 ch),
                        1x1 projection to `dim`, then `depth_ssm` SSD-Mamba blocks (~2.3 M params).
    backbone="custom" : the original 0.95 M-parameter from-scratch backbone (ablation).
    depth_ssm=0 gives the same network without SSM blocks (ablation "MicroPathMamba-CNN").
    """

    def __init__(self, num_classes=9, width=1.0, img_size=224, depth_ssm=2, d_state=16, heads=4,
                 backbone="mnv2", dim=160, pretrained=True):
        super().__init__()

        def ch(c):
            return max(8, int(math.ceil(c * width / 8) * 8))

        self.backbone = backbone
        if backbone == "mnv2":
            import timm   # MobileNetV2 inverted residuals + ReLU6: standard STM32N6 NPU workload
            self.features = timm.create_model("mobilenetv2_100.ra_in1k", pretrained=pretrained,
                                              features_only=True, out_indices=(4,))
            cin = self.features.feature_info.channels()[-1]             # 320
            self.proj = conv_bn_act(cin, dim, 1)
            cin = dim
        else:
            # (expand, out, repeats, stride) - MobileNetV2-like schedule, trained from scratch
            cfg = [(1, 16, 1, 1), (4, 24, 2, 2), (4, 40, 2, 2), (4, 80, 3, 2), (4, 112, 2, 1), (4, 160, 1, 2)]
            layers = [conv_bn_act(3, ch(16), 3, 2)]
            cin = ch(16)
            for e, c, n, s in cfg:
                for i in range(n):
                    layers.append(InvertedResidual(cin, ch(c), s if i == 0 else 1, e))
                    cin = ch(c)
            self.features = nn.Sequential(*layers)
            self.proj = nn.Identity()
        tokens = (img_size // 32) ** 2                                  # 49 for 224 input
        self.ssm = nn.Sequential(*[MambaBlock(cin, tokens, d_state, heads) for _ in range(depth_ssm)])
        self.head = nn.Sequential(conv_bn_act(cin, ch(320), 1), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.2), nn.Linear(ch(320), num_classes))

    def forward_features(self, x):
        f = self.features(x)
        if isinstance(f, (list, tuple)):
            f = f[-1]
        return self.ssm(self.proj(f))

    def forward(self, x):
        return self.head(self.forward_features(x))


def micropathmamba(num_classes=9, **kw):
    return MicroPathMamba(num_classes=num_classes, **kw)


if __name__ == "__main__":
    m = micropathmamba(pretrained=False).eval()
    n = sum(p.numel() for p in m.parameters())
    with torch.no_grad():
        out = m(torch.randn(2, 3, 224, 224))
    print(f"params: {n / 1e6:.2f} M | output: {tuple(out.shape)}")
