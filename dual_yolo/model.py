"""Two-backbone mean-fusion YOLO11 OBB baseline (Ultralytics 8.4.208)."""

from copy import deepcopy

import torch
from torch import nn
from ultralytics.nn.tasks import OBBModel


class DualBackboneOBB(OBBModel):
    """Independent RGB/thermal backbones; mean fusion at P3, P4, and P5.

    Packed input is [B, G, R, IR] because OpenCV reads four-channel TIFF
    as BGRA. The RGB branch is reordered to RGB before the pretrained model.
    """

    FUSION_LAYERS = (4, 6, 10)
    BACKBONE_END = 10

    def __init__(self, cfg="yolo11n-obb.yaml", nc=5, verbose=False):
        self.dual_ready = False
        super().__init__(cfg=cfg, ch=3, nc=nc, verbose=verbose)

    def attach_thermal_backbone(self):
        if self.dual_ready:
            return
        if len(self.model) != 24 or self.model[11].__class__.__name__ != "Upsample":
            raise ValueError("Expected the YOLO11n-OBB graph from Ultralytics 8.4.208")
        self.thermal_backbone = nn.ModuleList(deepcopy(list(self.model[: self.BACKBONE_END + 1])))
        first = self.thermal_backbone[0]
        old = first.conv
        new = nn.Conv2d(1, old.out_channels, old.kernel_size, old.stride,
                        old.padding, old.dilation, old.groups, bias=False)
        with torch.no_grad():
            # Equivalent to feeding a grayscale thermal image to all 3 RGB channels.
            new.weight.copy_(old.weight.sum(dim=1, keepdim=True))
        first.conv = new
        self.dual_ready = True
        self.yaml["ch"] = 4

    def _predict_once(self, x, profile=False, embed=None):
        if not self.dual_ready:
            return super()._predict_once(x, profile=profile, embed=embed)
        if x.ndim != 4 or x.shape[1] != 4:
            raise ValueError("Expected packed [B,4,H,W] RGB+thermal input")
        if profile or embed:
            raise NotImplementedError("Profiling and intermediate embeddings are not implemented")
        rgb = x[:, [2, 1, 0]]  # OpenCV BGRA -> RGB
        thermal = x[:, 3:4]
        saved = [None] * len(self.model)
        for i in range(self.BACKBONE_END + 1):
            rgb = self.model[i](rgb)
            thermal = self.thermal_backbone[i](thermal)
            if i in self.FUSION_LAYERS:
                fused = (rgb + thermal) * 0.5
                saved[i] = fused
        out = saved[self.BACKBONE_END]
        for m in list(self.model)[self.BACKBONE_END + 1:]:
            if m.f != -1:
                out = saved[m.f] if isinstance(m.f, int) else [out if j == -1 else saved[j] for j in m.f]
            out = m(out)
            if m.i in self.save:
                saved[m.i] = out
        return out
