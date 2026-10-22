"""Binary whole-metastasis trainers using native nnU-Net optimization.

Loss composition/deep supervision adapted from TIO-IKIM/Learning-to-Look-Closer
(Apache-2.0); see documentation/lossfunctions/proposals/01_cc_dice_ce.md.
"""

# ruff: noqa: N999
import json
import os
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import torch

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.instance_losses import (
    UPSTREAM_REVISION,
    BinaryCCDiceCELoss,
    global_dice_ce,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.connected_components import require_cc_backend


def validate_cc_dataset(dataset_json):
    labels = dataset_json.get("labels", {})
    if ("ignore" in labels or len(labels) != 2 or
            any(type(v) is not int for v in labels.values()) or
            set(labels.values()) != {0, 1} or labels.get("background") != 0):
        raise ValueError(
            "These trainers require binary class labels background=0, whole metastasis=1, "
            "not multiclass labels, overlapping regions, or ignore labels. Convert/reprocess "
            "the dataset: enhancing plus non-enhancing tumor is foreground; edema is background."
        )


def cc_deep_supervision_weights(count, is_ddp, compiled_network):
    weights = np.array([1 / (2 ** i) for i in range(count)], dtype=np.float64)
    if count > 1:
        weights[-1] = 1e-6 if is_ddp and not compiled_network else 0
    return (weights / weights.sum()).tolist()


class _BinaryInstanceTrainerMixin:
    """Only binary validation/loss construction; optimization stays in nnUNetTrainer."""
    include_global = False
    use_cc = True
    cc_loss_name = "cc_dice_ce"

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):  # noqa: B008 -- nnU-Net constructor contract
        validate_cc_dataset(dataset_json)
        super().__init__(plans, configuration, fold, dataset_json, device)
        if len(self.configuration_manager.patch_size) != 3:
            raise ValueError("CC trainers support 3D configurations only.")
        if self.label_manager.has_regions or self.label_manager.has_ignore_label or self.label_manager.num_segmentation_heads != 2:
            raise ValueError("These trainers require two mutually exclusive outputs without ignore labels.")

    def initialize(self):
        if not self.was_initialized and self.use_cc:
            # Check before allocating the network, optimizer, or compiling it.
            self.cc_backend_versions = require_cc_backend(self.device)
        super().initialize()

    def _build_loss(self):
        loss = BinaryCCDiceCELoss(
            include_global=self.include_global,
            batch_dice=self.configuration_manager.batch_dice, ddp=self.is_ddp,
        ) if self.use_cc else global_dice_ce(self.configuration_manager.batch_dice, self.is_ddp)
        weights = [1.0]
        if self.enable_deep_supervision:
            weights = cc_deep_supervision_weights(
                len(self._get_deep_supervision_scales()), self.is_ddp, self._do_i_compile()
            )
            loss = DeepSupervisionWrapper(loss, weights)
        versions = {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
                    **getattr(self, "cc_backend_versions", {})}
        for package in ("numpy", "nnunetv2", "dynamic-network-architectures"):
            try:
                versions[package] = version(package)
            except PackageNotFoundError:
                versions[package] = "not installed as distribution"
        resolved = {
            "trainer": type(self).__name__, "objective": self.cc_loss_name,
            "schema_version": 2,
            "labels": {"background": 0, "whole_metastasis": 1},
            "foreground_definition": "enhancing plus non-enhancing/necrotic tumor; excludes edema",
            "cc_weight": int(self.use_cc), "global_weight": int(self.include_global), "dice_smooth": 0,
            "deep_supervision_weights": weights, "connectivity": 26, "distance_units": "voxels",
            "empty_class": "CC: binary CE mean + 1", "cc_reduction": "components then local samples; DDP rank mean",
            "upstream_revision": UPSTREAM_REVISION, "versions": versions,
            "optimizer": type(self.optimizer).__name__ if self.optimizer is not None else "SGD",
            "initial_lr": self.initial_lr, "weight_decay": self.weight_decay,
            "momentum": 0.99, "nesterov": True, "lr_scheduler": "PolyLRScheduler",
            "lr_exponent": 0.9, "num_epochs": self.num_epochs,
            "iterations_per_epoch": self.num_iterations_per_epoch,
            "dataset": self.plans_manager.dataset_name, "plans": self.plans_manager.plans_name,
            "configuration": self.configuration_name, "fold": self.fold,
        }
        self.print_to_log_file("[CC] " + json.dumps(resolved))
        self.logger.update_config({"cc_loss": resolved})
        if self.local_rank == 0:
            with open(os.path.join(self.output_folder, "cc_loss_config.json"), "w") as stream:
                json.dump(resolved, stream, indent=2)
        return loss


class nnUNetTrainerCCDiceCE(_BinaryInstanceTrainerMixin, nnUNetTrainer):
    """Binary whole-metastasis CC Dice+CE, without a global term."""


class nnUNetTrainerGlobalCCDiceCE(_BinaryInstanceTrainerMixin, nnUNetTrainer):
    """Binary global Dice+CE plus CC Dice+CE, with coefficients 1:1."""

    include_global = True
    cc_loss_name = "global_cc_dice_ce"


class nnUNetTrainerDiceCEBaseline(_BinaryInstanceTrainerMixin, nnUNetTrainer):
    """Matched binary global Dice+CE baseline with smooth=0; no CuPy needed."""

    include_global = True
    use_cc = False
    cc_loss_name = "dice_ce_no_smooth"
