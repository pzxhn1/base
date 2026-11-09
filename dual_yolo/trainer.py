"""Ultralytics OBB trainer that installs the two-backbone model."""

from ultralytics.models.yolo.obb.train import OBBTrainer

from .model import DualBackboneOBB


class DualOBBTrainer(OBBTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = self.set_model_names_for_load(
            DualBackboneOBB(cfg=cfg or "yolo11n-obb.yaml", nc=self.data["nc"], verbose=verbose)
        )
        if weights:
            model.load(weights)
        model.attach_thermal_backbone()
        return model
