import numpy as np
import torch
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.focal_tversky import FocalTverskyLoss
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import nnUNetTrainerDA5
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import softmax_helper_dim1


class nnUNetTrainerDA5_PureFocalTversky(nnUNetTrainerDA5):
    """
    nnUNetTrainer utilizing Advanced Spatial Data Augmentations (DA5)
    and Pure Focal Tversky Loss (without Cross-Entropy).
    
    Default parameters (matching Abraham & Khan, ISBI 2019):
        alpha = 0.3 (False Positive weight)
        beta  = 0.7 (False Negative weight - prioritizes recall for small lesions)
        gamma = 1.33 (Focal exponent)
    """
    def _build_loss(self):
        loss = FocalTverskyLoss(
            apply_nonlin=torch.sigmoid if self.label_manager.has_regions else softmax_helper_dim1,
            alpha=0.3,
            beta=0.7,
            gamma=1.33,
            batch_dice=self.configuration_manager.batch_dice,
            do_bg=True if self.label_manager.has_regions else False,
            smooth=1e-5,
            ddp=self.is_ddp
        )

        if self._do_i_compile():
            loss = torch.compile(loss)

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


class nnUNetTrainerPureFocalTversky(nnUNetTrainer):
    """
    Standard baseline nnUNetTrainer with Pure Focal Tversky Loss.
    """
    def _build_loss(self):
        loss = FocalTverskyLoss(
            apply_nonlin=torch.sigmoid if self.label_manager.has_regions else softmax_helper_dim1,
            alpha=0.3,
            beta=0.7,
            gamma=1.33,
            batch_dice=self.configuration_manager.batch_dice,
            do_bg=True if self.label_manager.has_regions else False,
            smooth=1e-5,
            ddp=self.is_ddp
        )

        if self._do_i_compile():
            loss = torch.compile(loss)

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
