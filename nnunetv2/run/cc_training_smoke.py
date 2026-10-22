"""Exercise native initialization and real preprocessed batches in an isolated results root."""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from nnunetv2.run.run_training import get_trainer_from_args
from nnunetv2.training.nnUNetTrainer import nnUNetTrainer as base_module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--plans", default="nnUNetPlans")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--trainer", default="nnUNetTrainerGlobalCCDiceCE", choices=[
        "nnUNetTrainerDiceCEBaseline", "nnUNetTrainerCCDiceCE", "nnUNetTrainerGlobalCCDiceCE"])
    parser.add_argument("--output-dir", type=Path, required=True, help="New isolated smoke-results directory")
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    if args.steps < 2:
        parser.error("Use at least two steps to include a warmed-up step")
    if not torch.cuda.is_available():
        parser.error("This smoke test requires CUDA")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    base_module.nnUNet_results = str(output)
    os.environ["nnUNet_wandb_enabled"] = "false"
    trainer = get_trainer_from_args(args.dataset, args.configuration, args.fold,
                                   trainer_name=args.trainer, plans_identifier=args.plans)
    torch.cuda.reset_peak_memory_stats(trainer.device)
    trainer.initialize()
    train_loader = val_loader = None
    times, losses = [], []
    try:
        train_loader, val_loader = trainer.get_dataloaders()
        trainer.network.train()
        for _ in range(args.steps):
            batch = next(train_loader)
            torch.cuda.synchronize()
            start = perf_counter()
            loss = trainer.train_step(batch)["loss"]
            torch.cuda.synchronize()
            if not np.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            losses.append(float(loss))
            times.append(perf_counter() - start)
        trainer.network.eval()
        with torch.no_grad():
            validation = trainer.validation_step(next(val_loader))
        if not np.isfinite(validation["loss"]):
            raise RuntimeError("Nonfinite validation loss")
        checkpoint = Path(trainer.output_folder) / "checkpoint_latest.pth"
        trainer.save_checkpoint(str(checkpoint))
        trainer.load_checkpoint(str(checkpoint))
        trainer.network.train()
        resumed_loss = trainer.train_step(next(train_loader))["loss"]
        torch.cuda.synchronize()
        if not np.isfinite(resumed_loss):
            raise RuntimeError("Nonfinite training loss after checkpoint reload")
        report = {
            "trainer": args.trainer, "dataset": args.dataset, "fold": args.fold,
            "patch_size": trainer.configuration_manager.patch_size, "batch_size": trainer.batch_size,
            "losses": losses, "validation_loss": float(validation["loss"]),
            "step_seconds": times, "warm_step_mean_seconds": sum(times[1:]) / len(times[1:]),
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(trainer.device),
            "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(trainer.device),
            "checkpoint_reload": "passed",
            "resumed_training_loss": float(resumed_loss),
            "note": "Allocator statistics exclude some CUDA/context allocations; also monitor nvidia-smi.",
        }
        if trainer.use_cc:
            from nnunetv2.utilities.connected_components import _cupy_modules
            cp, _, _ = _cupy_modules()
            report["cupy_pool_reserved_bytes"] = cp.get_default_memory_pool().total_bytes()
        (output / "smoke_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        for loader in (train_loader, val_loader):
            if hasattr(loader, "_finish"):
                loader._finish()


if __name__ == "__main__":
    main()
