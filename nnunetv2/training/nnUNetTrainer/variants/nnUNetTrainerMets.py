import torch
from torch.optim import AdamW

from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerMets(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 3e-4
        self.weight_decay = 1e-4
        self.num_epochs = 500

    def configure_optimizers(self):
        optimizer = AdamW(self.network.parameters(),
                          lr=self.initial_lr,
                          weight_decay=self.weight_decay,
                          amsgrad=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler
