import numpy as np
import torch
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.focal_tversky import FTL_and_CE_loss, FTL_and_BCE_loss, FocalTverskyLoss
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import nnUNetTrainerDA5
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import softmax_helper_dim1


class nnUNetTrainerDA5_CompoundFocalTversky(nnUNetTrainerDA5):
    """
    nnUNetTrainer utilizing Advanced Spatial Data Augmentations (DA5)
    and Compound Focal Tversky Loss (1.0 * CE + 1.0 * FocalTversky).
    
    Default parameters:
        alpha = 0.3 (False Positive weight)
        beta  = 0.7 (False Negative weight - prioritizes recall for small lesions)
        gamma = 1.33 (Focal exponent)
    """
    def _build_loss(self):
        ftl_kwargs = {
            'alpha': 0.3,
            'beta': 0.7,
            'gamma': 1.33,
            'batch_dice': self.configuration_manager.batch_dice,
            'do_bg': True if self.label_manager.has_regions else False,
            'smooth': 1e-5,
            'ddp': self.is_ddp
        }

        if self.label_manager.has_regions:
            loss = FTL_and_BCE_loss(
                bce_kwargs={},
                ftl_kwargs=ftl_kwargs,
                weight_ce=1.0,
                weight_ftl=1.0,
                use_ignore_label=self.label_manager.ignore_label is not None
            )
        else:
            loss = FTL_and_CE_loss(
                ftl_kwargs=ftl_kwargs,
                ce_kwargs={},
                weight_ce=1.0,
                weight_ftl=1.0,
                ignore_label=self.label_manager.ignore_label
            )

        if self._do_i_compile():
            loss.ftl = torch.compile(loss.ftl)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss


# Aliases for convenience
class nnUNetTrainerDA5_FocalTversky(nnUNetTrainerDA5_CompoundFocalTversky):
    pass


class nnUNetTrainerCompoundFocalTversky(nnUNetTrainer):
    """
    Standard baseline nnUNetTrainer with Compound Focal Tversky Loss.
    """
    def _build_loss(self):
        ftl_kwargs = {
            'alpha': 0.3,
            'beta': 0.7,
            'gamma': 1.33,
            'batch_dice': self.configuration_manager.batch_dice,
            'do_bg': True if self.label_manager.has_regions else False,
            'smooth': 1e-5,
            'ddp': self.is_ddp
        }

        if self.label_manager.has_regions:
            loss = FTL_and_BCE_loss(
                bce_kwargs={},
                ftl_kwargs=ftl_kwargs,
                weight_ce=1.0,
                weight_ftl=1.0,
                use_ignore_label=self.label_manager.ignore_label is not None
            )
        else:
            loss = FTL_and_CE_loss(
                ftl_kwargs=ftl_kwargs,
                ce_kwargs={},
                weight_ce=1.0,
                weight_ftl=1.0,
                ignore_label=self.label_manager.ignore_label
            )

        if self._do_i_compile():
            loss.ftl = torch.compile(loss.ftl)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss
