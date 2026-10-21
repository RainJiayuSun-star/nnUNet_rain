"""Trainer discovery/configuration and checkpoint/inference plumbing on CPU.

These tests do not claim that a CPU trainer can run production CC geometry.
"""

import builtins
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from nnunetv2.run import run_training
from nnunetv2.training.nnUNetTrainer import nnUNetTrainer as base_module
from nnunetv2.training.nnUNetTrainer.variants.loss import (
    nnUNetTrainerInstanceLosses as trainers,
)
from nnunetv2.utilities.connected_components import require_cc_backend


def tiny_plans():
    return {
        "dataset_name": "Dataset999_CCTest", "plans_name": "CCTestPlans",
        "transpose_forward": [0, 1, 2], "transpose_backward": [0, 1, 2],
        "configurations": {"3d_fullres": {
            "data_identifier": "CCTestPlans_3d_fullres", "batch_size": 2,
            "patch_size": [16, 16, 16], "batch_dice": False,
            "architecture": {
                "network_class_name": "dynamic_network_architectures.architectures.unet.PlainConvUNet",
                "arch_kwargs": {
                    "n_stages": 3, "features_per_stage": [4, 8, 16],
                    "conv_op": "torch.nn.modules.conv.Conv3d",
                    "kernel_sizes": [[3, 3, 3]] * 3, "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2]],
                    "n_conv_per_stage": [1, 1, 1], "n_conv_per_stage_decoder": [1, 1],
                    "conv_bias": True, "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
                    "norm_op_kwargs": {"eps": 1e-5, "affine": True},
                    "dropout_op": "torch.nn.modules.dropout.Dropout3d", "dropout_op_kwargs": {"p": 0.3},
                    "nonlin": "torch.nn.LeakyReLU", "nonlin_kwargs": {"inplace": True},
                },
                "_kw_requires_import": ["conv_op", "norm_op", "dropout_op", "nonlin"],
            },
        }},
    }


@pytest.fixture
def training_files(tmp_path, monkeypatch):
    monkeypatch.setenv("nnUNet_wandb_enabled", "false")
    monkeypatch.setenv("nnUNet_compile", "false")
    preprocessed = tmp_path / "preprocessed"
    dataset = preprocessed / "Dataset999_CCTest"
    dataset.mkdir(parents=True)
    plans = tiny_plans()
    labels = {"labels": {"background": 0, "tumor_core": 1, "edema": 2},
              "channel_names": {"0": "FLAIR", "1": "T1post"}, "file_ending": ".nii.gz"}
    (dataset / "CCTestPlans.json").write_text(json.dumps(plans))
    (dataset / "dataset.json").write_text(json.dumps(labels))
    monkeypatch.setattr(run_training, "nnUNet_preprocessed", str(preprocessed))
    monkeypatch.setattr(base_module, "nnUNet_preprocessed", str(preprocessed))
    monkeypatch.setattr(base_module, "nnUNet_results", str(tmp_path / "results"))
    preset = Path(__file__).resolve().parents[2] / "configs/cc_multiclass.yaml"
    monkeypatch.setenv("NNUNET_CONFIG_PATH", str(preset))
    return plans, labels


def make_trainer(name):
    return run_training.get_trainer_from_args(
        "Dataset999_CCTest", "3d_fullres", 0, trainer_name=name,
        plans_identifier="CCTestPlans", device=torch.device("cpu"),
    )


@pytest.mark.parametrize("name", ["nnUNetTrainerCCDiceCE", "nnUNetTrainerGlobalCCDiceCE"])
def test_discovery_config_metadata_and_backend_guard(training_files, name):
    trainer = make_trainer(name)
    assert type(trainer).__name__ == name
    assert trainer.optimizer_type == "AdamW" and trainer.initial_lr == 0.0002
    assert trainer.num_epochs == 300 and trainer.dropout_p == 0.3
    assert trainer.region_weights is None and trainer.experiment_name is None
    with pytest.raises(RuntimeError, match="CUDA"):
        trainer.initialize()
    assert trainer.network is None  # Fail before allocating a network.
    loss = trainer._build_loss()  # Construction is safe; forward needs CUDA.
    assert (loss.loss.global_loss is not None) == ("Global" in name)
    metadata = json.loads((Path(trainer.output_folder) / "cc_loss_config.json").read_text())
    assert metadata["class_weights"] == [0.5, 0.5]
    assert metadata["dice_smooth"] == 0
    assert metadata["global_weight"] == int("Global" in name)


@pytest.mark.parametrize("labels", [
    {"background": 0, "lesion": 1},
    {"background": 0, "WT": [1, 2, 3], "TC": [2, 3], "ET": 3},
    {"background": 0, "core": 1, "edema": 2, "ignore": 3},
    {"background": 0, "core": 1, "edema": 3},
])
def test_reject_invalid_labels(labels):
    with pytest.raises(ValueError, match="require class labels"):
        trainers.validate_cc_dataset({"labels": labels})


def test_reject_2d(training_files):
    plans, labels = copy.deepcopy(training_files)
    plans["continue_training"] = False
    plans["configurations"]["3d_fullres"]["patch_size"] = [16, 16]
    with pytest.raises(ValueError, match="3D"):
        trainers.nnUNetTrainerCCDiceCE(plans, "3d_fullres", 0, labels, torch.device("cpu"))


@pytest.mark.parametrize("loss_name", ["cc_dice_ce", "global_cc_dice_ce"])
def test_config_conflicts(tmp_path, loss_name):
    path = tmp_path / "config.yaml"
    for incompatible in ("focal_tversky", "misspelled", "global_cc_dice_ce" if loss_name == "cc_dice_ce" else "cc_dice_ce"):
        path.write_text(f"loss_type: {incompatible}\n")
        with pytest.raises(ValueError, match="conflicting"):
            trainers.read_cc_config(path, loss_name)
    for allowed in (loss_name, "dice_ce"):
        path.write_text(f"loss_type: {allowed}\n")
        assert trainers.read_cc_config(path, loss_name)["loss_type"] == allowed
    path.write_text("- not a mapping\n")
    with pytest.raises(TypeError, match="mapping"):
        trainers.read_cc_config(path, loss_name)


def test_scale_weights():
    np.testing.assert_allclose(trainers.cc_deep_supervision_weights(4, False, False), [4/7, 2/7, 1/7, 0])
    assert trainers.cc_deep_supervision_weights(4, True, False)[-1] > 0
    assert trainers.cc_deep_supervision_weights(4, True, True)[-1] == 0
    assert trainers.cc_deep_supervision_weights(1, False, False) == [1]


def test_ordinary_trainers_discover_without_cupy(training_files, monkeypatch):
    original_import = builtins.__import__

    def no_cupy(name, *args, **kwargs):
        if name.startswith(("cupy", "cupyx")):
            raise AssertionError("Discovery must not import CuPy")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_cupy)
    for name in ("nnUNetTrainer", "nnUNetTrainerMetsConfigurable", "nnUNetTrainer_CompoundFocalTversky"):
        assert type(make_trainer(name)).__name__ == name


@pytest.mark.parametrize("name", ["nnUNetTrainerCCDiceCE", "nnUNetTrainerGlobalCCDiceCE"])
def test_checkpoint_and_inference_without_yaml_or_cupy(training_files, monkeypatch, name):
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    trainer = make_trainer(name)
    config = trainer.configuration_manager
    # CPU architecture/checkpoint plumbing only; do not bypass production CUDA
    # checks to pretend to run a CC training iteration.
    trainer.network = trainer.build_network_architecture(
        config.network_arch_class_name, config.network_arch_init_kwargs,
        config.network_arch_init_kwargs_req_import, 2, 3, enable_deep_supervision=False,
    )
    trainer.optimizer, trainer.lr_scheduler = trainer.configure_optimizers()
    trainer.was_initialized = True
    checkpoint = Path(trainer.output_folder) / "checkpoint_final.pth"
    trainer.save_checkpoint(str(checkpoint))
    original = {k: v.clone() for k, v in trainer.network.state_dict().items()}
    with torch.no_grad():
        next(trainer.network.parameters()).add_(1)
    trainer.load_checkpoint(str(checkpoint))
    for k, v in trainer.network.state_dict().items():
        torch.testing.assert_close(v, original[k])
    root = Path(trainer.output_folder_base)
    (root / "plans.json").write_text(json.dumps(training_files[0]))
    (root / "dataset.json").write_text(json.dumps(training_files[1]))
    monkeypatch.setenv("NNUNET_CONFIG_PATH", str(root / "nonexistent.yaml"))
    original_import = builtins.__import__

    def no_cupy(module, *args, **kwargs):
        if module.startswith(("cupy", "cupyx")):
            raise AssertionError("Inference must not import CuPy")
        return original_import(module, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_cupy)
    # Inference must use the class's architecture builder, not instantiate its
    # YAML-reading/training-backend-checking constructor.
    def no_constructor(*args, **kwargs):
        raise AssertionError("Prediction must not instantiate a trainer")

    monkeypatch.setattr(trainers._TwoLabelCCTrainer, "__init__", no_constructor)
    predictor = nnUNetPredictor(device=torch.device("cpu"), perform_everything_on_device=False)
    predictor.initialize_from_trained_model_folder(str(root), use_folds=(0,))
    x = torch.randn(1, 2, 16, 16, 16)
    trainer.network.eval()
    predictor.network.eval()
    with torch.no_grad():
        torch.testing.assert_close(predictor.network(x), trainer.network(x))


def test_backend_error_is_actionable():
    with pytest.raises(RuntimeError, match="CPU/MPS"):
        require_cc_backend(torch.device("cpu"))
