# Subclass from nnUNetTrainerDA5, adjust the oversample_foreground_percent, for new folder spawn for new trainings
import torch
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import nnUNetTrainerDA5

class nnUNetTrainerDA5_OS50(nnUNetTrainerDA5):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # Adjust the oversample_foreground_percent here (default is 0.33)
        self.oversample_foreground_percent = 0.50