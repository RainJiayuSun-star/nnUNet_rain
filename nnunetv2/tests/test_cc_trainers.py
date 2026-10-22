"""Native trainer integration; production CC geometry is tested on CUDA only."""
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

TRAINERS = ("nnUNetTrainerDiceCEBaseline", "nnUNetTrainerCCDiceCE", "nnUNetTrainerGlobalCCDiceCE")


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
                    "dropout_op": None, "dropout_op_kwargs": None,
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
    arrays = dataset / "CCTestPlans_3d_fullres"
    arrays.mkdir()
    np.savez_compressed(arrays / "synthetic.npz", data=np.zeros((2, 16, 16, 16), dtype=np.float32),
                        seg=np.zeros((1, 16, 16, 16), dtype=np.int16))
    plans = tiny_plans()
    labels = {"labels": {"background": 0, "whole_metastasis": 1},
              "channel_names": {"0": "FLAIR", "1": "T1post"}, "file_ending": ".nii.gz"}
    (dataset / "CCTestPlans.json").write_text(json.dumps(plans))
    (dataset / "dataset.json").write_text(json.dumps(labels))
    monkeypatch.setattr(run_training, "nnUNet_preprocessed", str(preprocessed))
    monkeypatch.setattr(base_module, "nnUNet_preprocessed", str(preprocessed))
    monkeypatch.setattr(base_module, "nnUNet_results", str(tmp_path / "results"))
    config = tmp_path / "train_config.yaml"
    config.write_text("optimizer: AdamW\ninitial_lr: 7\nnum_epochs: 2\ndropout_p: 0.3\n")
    monkeypatch.setenv("NNUNET_CONFIG_PATH", str(config))
    monkeypatch.chdir(tmp_path)
    return plans, labels


def make_trainer(name, device="cpu"):
    return run_training.get_trainer_from_args(
        "Dataset999_CCTest", "3d_fullres", 0, trainer_name=name,
        plans_identifier="CCTestPlans", device=torch.device(device),
    )


@pytest.mark.parametrize("name", TRAINERS)
def test_discovery_native_defaults_and_backend_guard(training_files, name):
    trainer = make_trainer(name)
    assert type(trainer).__name__ == name
    assert trainer.initial_lr == 0.01 and trainer.weight_decay == 3e-5
    assert trainer.num_epochs == 1000 and trainer.num_iterations_per_epoch == 250
    assert trainer.configuration_manager.network_arch_init_kwargs["dropout_op_kwargs"] is None
    assert not hasattr(trainer, "config_path")
    if trainer.use_cc:
        with pytest.raises(RuntimeError, match="CUDA"):
            trainer.initialize()
        assert trainer.network is None
    else:
        trainer.initialize()
        assert isinstance(trainer.optimizer, torch.optim.SGD)
        assert trainer.optimizer.defaults["momentum"] == 0.99
        assert trainer.optimizer.defaults["nesterov"]
        assert trainer.lr_scheduler.max_steps == 1000
    loss = trainer._build_loss()
    inner = loss.loss
    if trainer.use_cc:
        assert (inner.global_loss is not None) == trainer.include_global
    else:
        assert inner.dc.smooth == 0 and not inner.dc.do_bg
    metadata = json.loads((Path(trainer.output_folder) / "cc_loss_config.json").read_text())
    assert metadata["labels"] == {"background": 0, "whole_metastasis": 1}
    assert metadata["optimizer"] == "SGD" and metadata["dice_smooth"] == 0
    assert metadata["cc_weight"] == int(trainer.use_cc)
    assert metadata["global_weight"] == int(trainer.include_global)


@pytest.mark.parametrize("labels", [
    {"background": 0, "core": 1, "edema": 2},
    {"background": 0, "WT": [1, 2, 3], "TC": [2, 3], "ET": 3},
    {"background": 0, "lesion": 1, "ignore": 2},
    {"background": 0, "lesion": 2},
    {"background": 0, "lesion": True},
])
def test_reject_invalid_labels(labels):
    with pytest.raises(ValueError, match="require binary"):
        trainers.validate_cc_dataset({"labels": labels})


def test_reject_2d(training_files):
    plans, labels = copy.deepcopy(training_files)
    plans["continue_training"] = False
    plans["configurations"]["3d_fullres"]["patch_size"] = [16, 16]
    with pytest.raises(ValueError, match="3D"):
        trainers.nnUNetTrainerCCDiceCE(plans, "3d_fullres", 0, labels, torch.device("cpu"))


def test_scale_weights():
    np.testing.assert_allclose(trainers.cc_deep_supervision_weights(4, False, False), [4/7, 2/7, 1/7, 0])
    assert trainers.cc_deep_supervision_weights(4, True, False)[-1] > 0
    assert trainers.cc_deep_supervision_weights(4, True, True)[-1] == 0
    assert trainers.cc_deep_supervision_weights(1, False, False) == [1]


def forbid_cupy(monkeypatch):
    original_import = builtins.__import__

    def no_cupy(name, *args, **kwargs):
        if name.startswith(("cupy", "cupyx")):
            raise AssertionError("Must not import CuPy")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_cupy)


def training_step(trainer):
    batch = {"data": torch.randn(2, 2, 16, 16, 16),
             "target": [torch.zeros(2, 1, n, n, n, dtype=torch.int16) for n in (16, 8)]}
    batch["target"][0][0, 0, 2:4, 2:4, 2:4] = 1
    trainer.network.train()
    assert np.isfinite(trainer.train_step(batch)["loss"])
    trainer.network.eval()
    with torch.no_grad():
        result = trainer.validation_step(batch)
    assert np.isfinite(result["loss"])
    assert result["tp_hard"].shape == (1,)


def test_baseline_steps_and_resume_without_cupy(training_files, monkeypatch):
    forbid_cupy(monkeypatch)
    trainer = make_trainer("nnUNetTrainerDiceCEBaseline")
    trainer.initialize()
    training_step(trainer)
    trainer.current_epoch = 10
    trainer.lr_scheduler.step(10)
    checkpoint = Path(trainer.output_folder) / "checkpoint_latest.pth"
    trainer.save_checkpoint(str(checkpoint))
    resumed = make_trainer("nnUNetTrainerDiceCEBaseline")
    resumed.load_checkpoint(str(checkpoint))
    assert resumed.current_epoch == 11
    assert resumed.optimizer.state_dict()["state"]
    for k, value in trainer.network.state_dict().items():
        torch.testing.assert_close(resumed.network.state_dict()[k], value)
    training_step(resumed)


@pytest.mark.parametrize("name", TRAINERS[1:])
def test_cc_native_integration_with_cpu_geometry_stub(training_files, monkeypatch, name):
    # Exercises native initialization/optimizer/checkpoint plumbing, NOT CUDA acceptance.
    from nnunetv2.tests.test_cc_losses import reference_voronoi
    from nnunetv2.training.loss import instance_losses

    monkeypatch.setattr(trainers, "require_cc_backend", lambda device: {"geometry": "CPU test stub"})
    monkeypatch.setattr(instance_losses, "compute_voronoi", reference_voronoi)
    trainer = make_trainer(name)
    assert base_module.nnUNetTrainer in type(trainer).__bases__
    trainer.initialize()
    assert isinstance(trainer.optimizer, torch.optim.SGD)
    assert trainer.optimizer.defaults["lr"] == 0.01
    assert trainer.optimizer.defaults["weight_decay"] == 3e-5
    assert trainer.optimizer.defaults["momentum"] == 0.99
    assert trainer.optimizer.defaults["nesterov"]
    assert trainer.lr_scheduler.max_steps == 1000
    training_step(trainer)
    path = str(Path(trainer.output_folder) / "checkpoint_latest.pth")
    trainer.save_checkpoint(path)
    resumed = make_trainer(name)
    resumed.load_checkpoint(path)
    assert resumed.optimizer.state_dict()["state"]
    training_step(resumed)


@pytest.mark.parametrize("name", TRAINERS)
def test_prediction_without_yaml_or_cupy(training_files, monkeypatch, name):
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    forbid_cupy(monkeypatch)
    trainer = make_trainer(name)
    config = trainer.configuration_manager
    # Checkpoint format/architecture on CPU; this is not CC training.
    trainer.network = trainer.build_network_architecture(
        config.network_arch_class_name, config.network_arch_init_kwargs,
        config.network_arch_init_kwargs_req_import, 2, 2, enable_deep_supervision=False,
    )
    trainer.optimizer, trainer.lr_scheduler = trainer.configure_optimizers()
    trainer.was_initialized = True
    trainer.save_checkpoint(str(Path(trainer.output_folder) / "checkpoint_final.pth"))
    root = Path(trainer.output_folder_base)
    (root / "plans.json").write_text(json.dumps(training_files[0]))
    (root / "dataset.json").write_text(json.dumps(training_files[1]))

    def no_constructor(*args, **kwargs):
        raise AssertionError("Prediction must not instantiate a trainer")

    monkeypatch.setattr(trainers._BinaryInstanceTrainerMixin, "__init__", no_constructor)
    predictor = nnUNetPredictor(device=torch.device("cpu"), perform_everything_on_device=False)
    predictor.initialize_from_trained_model_folder(str(root), use_folds=(0,))
    trainer.network.eval()
    predictor.network.eval()
    with torch.no_grad():
        x = torch.randn(1, 2, 16, 16, 16)
        torch.testing.assert_close(predictor.network(x), trainer.network(x))
        assert predictor.network(x).shape[1] == 2


def test_ordinary_trainers_discover_without_cupy(training_files, monkeypatch):
    forbid_cupy(monkeypatch)
    for name in ("nnUNetTrainer", "nnUNetTrainerDA5", "nnUNetTrainer_CompoundFocalTversky"):
        assert type(make_trainer(name)).__name__ == name
