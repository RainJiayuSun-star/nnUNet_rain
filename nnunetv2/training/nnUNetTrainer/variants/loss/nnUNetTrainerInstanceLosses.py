"""Project multiclass CC trainers, NOT binary upstream trainer reproductions.

Loss composition/deep supervision adapted from TIO-IKIM/Learning-to-Look-Closer
(Apache-2.0); see documentation/lossfunctions/proposals/01_cc_dice_ce.md.
"""

# ruff: noqa: N999
import json
import os
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import torch
import yaml

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.instance_losses import (
    UPSTREAM_REVISION,
    TwoLabelCCDiceCELoss,
)
from nnunetv2.training.nnUNetTrainer.variants.nnUNetTrainerMetsConfigurable import (
    nnUNetTrainerMetsConfigurable,
)
from nnunetv2.utilities.connected_components import require_cc_backend


def validate_cc_dataset(dataset_json):
    labels = dataset_json.get("labels", {})
    if ("ignore" in labels or not labels or
            any(type(v) is not int for v in labels.values()) or
            set(labels.values()) != {0, 1, 2} or labels.get("background") != 0):
        raise ValueError(
            "These CC trainers require class labels background=0, tumor core=1, edema=2, "
            "not binary labels, overlapping regions, or ignore labels. Verify/reprocess the dataset; "
            "old WT/TC/ET preprocessing is incompatible."
        )


def read_cc_config(path, loss_name):
    if not os.path.isfile(path):
        return {}  # Preserve the configurable trainer's missing-file defaults.
    with open(path) as stream:
        config = yaml.safe_load(stream)
    config = {} if config is None else config
    if not isinstance(config, dict):
        raise TypeError("CC training YAML must be a mapping.")
    # dice_ce is the parent trainer's legacy default, not a CC override.
    if config.get("loss_type") not in (None, "dice_ce", loss_name):
        raise ValueError(f"Trainer selects {loss_name}; conflicting YAML loss_type={config['loss_type']!r}.")
    return config


def cc_deep_supervision_weights(count, is_ddp, compiled_network):
    weights = np.array([1 / (2 ** i) for i in range(count)], dtype=np.float64)
    if count > 1:
        weights[-1] = 1e-6 if is_ddp and not compiled_network else 0
    return (weights / weights.sum()).tolist()


class _TwoLabelCCTrainer(nnUNetTrainerMetsConfigurable):
    include_global = False
    cc_loss_name = "cc_dice_ce"

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):  # noqa: B008 -- nnU-Net constructor contract
        validate_cc_dataset(dataset_json)
        read_cc_config(os.environ.get("NNUNET_CONFIG_PATH", "train_config.yaml"), self.cc_loss_name)
        super().__init__(plans, configuration, fold, dataset_json, device)
        if len(self.configuration_manager.patch_size) != 3:
            raise ValueError("CC trainers support 3D configurations only.")
        if self.label_manager.has_regions or self.label_manager.has_ignore_label or self.label_manager.num_segmentation_heads != 3:
            raise ValueError("CC trainers require three mutually exclusive class outputs without ignore labels.")
        self.loss_type = self.cc_loss_name
        if self.region_weights is not None:
            self.print_to_log_file("WARNING: region_weights do not apply to class-based CC training and are ignored.")

    def initialize(self):
        if not self.was_initialized:
            # Check before allocating the network, optimizer, or compiling it.
            self.cc_backend_versions = require_cc_backend(self.device)
        super().initialize()

    def _build_loss(self):
        loss = TwoLabelCCDiceCELoss(
            include_global=self.include_global,
            batch_dice=self.configuration_manager.batch_dice, ddp=self.is_ddp,
        )
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
            "labels": {"background": 0, "core": 1, "edema": 2}, "class_weights": [0.5, 0.5],
            "cc_weight": 1, "global_weight": int(self.include_global), "dice_smooth": 0,
            "deep_supervision_weights": weights, "connectivity": 26, "distance_units": "voxels",
            "empty_class": "binary CE mean + 1", "cc_reduction": "components then local samples then classes; DDP rank mean",
            "upstream_revision": UPSTREAM_REVISION, "versions": versions,
            "optimizer": self.optimizer_type, "initial_lr": self.initial_lr,
            "weight_decay": self.weight_decay, "num_epochs": self.num_epochs, "dropout_p": self.dropout_p,
            "dataset": self.plans_manager.dataset_name, "plans": self.plans_manager.plans_name,
            "configuration": self.configuration_name, "fold": self.fold,
        }
        self.print_to_log_file("[CC] " + json.dumps(resolved))
        self.logger.update_config({"cc_loss": resolved})
        if self.local_rank == 0:
            with open(os.path.join(self.output_folder, "cc_loss_config.json"), "w") as stream:
                json.dump(resolved, stream, indent=2)
        return loss


class nnUNetTrainerCCDiceCE(_TwoLabelCCTrainer):
    """Equal core/edema CC Dice+CE, without a global term."""


class nnUNetTrainerGlobalCCDiceCE(_TwoLabelCCTrainer):
    """The identical core/edema CC term plus global three-class Dice+CE."""

    include_global = True
    cc_loss_name = "global_cc_dice_ce"
