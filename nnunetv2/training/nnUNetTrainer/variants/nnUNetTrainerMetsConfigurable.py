import os
import shutil
import yaml
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper


class WeightedRegionLossWrapper(nn.Module):
    def __init__(self, base_loss: nn.Module, weights: list):
        super().__init__()
        self.base_loss = base_loss
        # Normalize weights so they average to 1.0 (maintaining loss scale)
        weights_arr = np.array(weights, dtype=np.float32)
        weights_arr = weights_arr / weights_arr.mean()
        self.register_buffer("weights", torch.from_numpy(weights_arr))

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        num_regions = net_output.shape[1]
        assert len(self.weights) == num_regions, \
            f"Expected {num_regions} regional weights, but config specified {len(self.weights)}"

        total_loss = 0.0
        for r in range(num_regions):
            region_output = net_output[:, r:r+1]
            
            # Handle ignore label channel if present (appended as the last channel of target)
            if target.shape[1] > num_regions:
                region_target = torch.cat([target[:, r:r+1], target[:, -1:]], dim=1)
            else:
                region_target = target[:, r:r+1]

            r_loss = self.base_loss(region_output, region_target)
            total_loss += self.weights[r] * r_loss

        return total_loss / num_regions


class nnUNetTrainerMetsConfigurable(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        # Path to config file (defaults to train_config.yaml in the working directory)
        self.config_path = os.environ.get("NNUNET_CONFIG_PATH", "train_config.yaml")

        # Default fallback values (matching nnUNetTrainerMets)
        self.experiment_name = None
        self.optimizer_type = "AdamW"
        self.initial_lr = 3e-4
        self.weight_decay = 1e-4
        self.num_epochs = 500
        self.dropout_p = None
        self.region_weights = None

        # Load YAML configuration if it exists
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    cfg = yaml.safe_load(f)

                if cfg is not None:
                    self.experiment_name = cfg.get("experiment_name", self.experiment_name)
                    self.optimizer_type = cfg.get("optimizer", self.optimizer_type)
                    self.initial_lr = float(cfg.get("initial_lr", self.initial_lr))
                    self.weight_decay = float(cfg.get("weight_decay", self.weight_decay))
                    self.num_epochs = int(cfg.get("num_epochs", self.num_epochs))
                    self.dropout_p = cfg.get("dropout_p", self.dropout_p)
                    self.region_weights = cfg.get("region_weights", self.region_weights)
            except Exception as e:
                print(f"Error parsing config file inside constructor: {e}")

        # Update output directory paths dynamically if experiment_name is set
        if self.experiment_name is not None:
            new_base_folder = f"{self.__class__.__name__}_{self.experiment_name}__{self.plans_manager.plans_name}__{configuration}"
            self.output_folder_base = os.path.join(os.path.dirname(self.output_folder_base), new_base_folder)
            self.output_folder = os.path.join(self.output_folder_base, f'fold_{fold}')
            
            # Recreate output folder
            os.makedirs(self.output_folder, exist_ok=True)
            
            # Re-initialize log file path and MetaLogger
            log_filename = os.path.basename(self.log_file)
            self.log_file = os.path.join(self.output_folder, log_filename)
            
            # Infer continue_training by checking if checkpoint files already exist in this folder
            checkpoint_exists = os.path.exists(os.path.join(self.output_folder, "checkpoint_final.pth")) or \
                                os.path.exists(os.path.join(self.output_folder, "checkpoint_best.pth"))
            
            from nnunetv2.training.logging.nnunet_logger import MetaLogger
            self.logger = MetaLogger(self.output_folder, checkpoint_exists)

        # Log configuration variables to the redirected log file and copy config file
        if os.path.exists(self.config_path):
            self.print_to_log_file(f"--- [CONFIG] Loading training parameters from: {self.config_path} ---")
            if self.experiment_name is not None:
                self.print_to_log_file(f"--- [CONFIG] Custom Experiment Directory: {self.output_folder} ---")
            try:
                # Copy configuration to output folder for logging/verification
                os.makedirs(self.output_folder, exist_ok=True)
                shutil.copy(self.config_path, os.path.join(self.output_folder, "train_config.yaml"))
                self.print_to_log_file(f"--- [CONFIG] Successfully loaded and backed up config file ---")
            except Exception as e:
                self.print_to_log_file(f"--- [CONFIG] Error backing up config file: {e} ---")
        else:
            self.print_to_log_file(f"--- [CONFIG] No config file found at {self.config_path}. Using defaults. ---")


    def initialize(self):
        if not self.was_initialized:
            # Overwrite dropout probability in plans before building the model architecture
            if self.dropout_p is not None:
                arch_kwargs = self.configuration_manager.network_arch_init_kwargs
                if 'dropout_op_kwargs' in arch_kwargs:
                    arch_kwargs['dropout_op_kwargs']['p'] = self.dropout_p
                    self.print_to_log_file(f"--- [CONFIG] Overriding architecture dropout to: {self.dropout_p} ---")
        super().initialize()
        
        # Log custom hyperparameters to W&B / MetaLogger configuration metadata
        if hasattr(self, 'logger') and self.logger is not None:
            custom_hparas = {
                "experiment_name": self.experiment_name,
                "dropout_p": self.dropout_p,
                "region_weights": self.region_weights,
                "optimizer_type": self.optimizer_type
            }
            self.logger.update_config({"custom_hparas": custom_hparas})


    def configure_optimizers(self):
        self.print_to_log_file(f"--- [CONFIG] Initializing {self.optimizer_type} optimizer "
                               f"(LR: {self.initial_lr}, WD: {self.weight_decay}, Epochs: {self.num_epochs}) ---")
        if self.optimizer_type.upper() == "ADAMW":
            optimizer = AdamW(self.network.parameters(), lr=self.initial_lr,
                              weight_decay=self.weight_decay, amsgrad=True)
        else:
            optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, 
                                        weight_decay=self.weight_decay, momentum=0.99, nesterov=True)

        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler

    def _build_loss(self):
        loss = super()._build_loss()

        # If region-based training is active and weights are defined, wrap the base loss
        if self.label_manager.has_regions and self.region_weights is not None:
            self.print_to_log_file(f"--- [CONFIG] Wrapping loss with custom region weights: {self.region_weights} ---")
            
            # If DeepSupervision is active, the loss is already wrapped in DeepSupervisionWrapper.
            # We must wrap the inner loss instead of the DeepSupervisionWrapper.
            if isinstance(loss, DeepSupervisionWrapper):
                loss.loss = WeightedRegionLossWrapper(loss.loss, self.region_weights)
            else:
                loss = WeightedRegionLossWrapper(loss, self.region_weights)

        return loss
