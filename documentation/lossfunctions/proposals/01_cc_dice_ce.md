# Two-label CC-DiceCE integration

Status: implemented; CPU checks passed. CUDA, AMP-on-CUDA, compiled-network-on-CUDA, DDP, and representative training performance remain GPU acceptance gates. No dataset conversion or model training was performed.

## Purpose and provenance

Expose exactly two custom trainers through the normal `nnUNetv2_train ... -tr` interface:

- `nnUNetTrainerCCDiceCE`: equal-weight tumor-core and edema component losses.
- `nnUNetTrainerGlobalCCDiceCE`: the identical component objective plus global three-class Dice+CE.

Both inherit the project's YAML-configurable trainer, not upstream's default SGD training recipe. Existing trainers and their loss behavior are unchanged. Blob trainers and cascade design are outside this integration.

Source: [TIO-IKIM/Learning-to-Look-Closer](https://github.com/TIO-IKIM/Learning-to-Look-Closer), pinned local commit `e5013bdb560eee544a94e1f068cb5cb4d003a51c`, especially `instance_losses.py`, `connected_components.py`, and `nnUNetTrainerInstanceLosses.py`. The original code is Apache-2.0; this checkout's [LICENSE](../../../LICENSE) contains that license. Adapted source files retain attribution and modification notices. See the [paper](https://arxiv.org/abs/2511.17146), Eqs. 1–3 and §3.1, for the component partition and global-plus-instance composition.

**These names identify project multiclass adaptations, not unchanged upstream binary trainers.** Upstream accepts two output channels. Our trainers require three output channels and separate class labels `{background: 0, tumor_core: 1, edema: 2}`. Input modalities remain whatever the verified plan specifies (the intended project inputs are FLAIR and T1post). Stored labels are not rewritten.

## Exact objective

At each active scale, logits are `[B,3,D,H,W]` and targets `[B,1,D,H,W]`. The class-specific binary views are:

```text
core:  [logsumexp(z_background, z_edema), z_core], target = (y == 1)
edema: [logsumexp(z_background, z_core), z_edema], target = (y == 2)
```

The binary foreground probability equals the original three-class probability exactly. Gradients flow through all logits. There is no additional network head, no independent sigmoid prediction, and no renormalization after dropping a channel.

For each foreground class separately, identify 26-connected GT components and assign every voxel to its nearest component in Euclidean **voxel coordinates**. Do not combine core and edema, use physical-spacing distances, or reuse/interpolate component maps across scales. Component maps are computed after augmentation from the actual target at that scale.

For a nonempty class in a sample, each component's Voronoi region `R_k` contributes:

```text
D_k = 1 - 2 * sum_Rk(p*t) / max(sum_Rk(p) + sum_Rk(t), 1e-7)
CE_k = mean_Rk(-log(max(p_binary[target], 1e-7)))
L_class_sample = mean_k(D_k + CE_k)
```

Average component losses within each sample, then samples within each local batch. Average the resulting core and edema terms equally:

```text
L_CC = 0.5 * (L_CC_core + L_CC_edema)
CCDiceCE       = sum_s w_s * L_CC(z_s, y_s)
GlobalCCDiceCE = sum_s w_s * [L_CC(z_s, y_s) + L_global(z_s, y_s)]
```

`L_global` is the existing `DC_and_CE_loss` with CE and Dice weights 1, additive Dice smoothing 0, `do_bg=False`, and the plan's `batch_dice` setting. CE supervises all three classes; Dice averages core and edema. Global Dice returns negative Dice, whereas component Dice uses `1 - Dice`; retain these offsets when interpreting training-loss values. The global/combined-CC coefficients are fixed at 1 for this comparison.

Deep-supervision weights follow nnU-Net: `1/2**s`, lowest output zero except `1e-6` under DDP without network compilation, then normalized. A single supervised output receives weight 1. Zero-weight outputs do not compute geometry. No extra finest-only term is added.

For an absent class, skip the EDT and return whole-patch binary CE plus the upstream constant 1. Include that class in the equal class average; it must still penalize false positives. This also covers lesions disappearing at coarse scales. Full masks and crop-truncated components use the same voxel-domain objective.

Under DDP, CC retains upstream local-sample averaging followed by DDP's rank averaging; do not describe unequal per-rank batch sizes as an exact global sample mean. The global Dice retains nnU-Net's configured batch-Dice/DDP behavior. Use equal per-rank batch sizes for tightly controlled comparisons.

## Implementation and compatibility

- [Loss module](../../../nnunetv2/training/loss/instance_losses.py): differentiable binary adapters, vectorized per-component scatter reductions, and optional global term.
- [Geometry](../../../nnunetv2/utilities/connected_components.py): lazy CuPy import, explicit CUDA device/current-stream interoperability, empty-mask guard, no autograd.
- [Trainers](../../../nnunetv2/training/nnUNetTrainer/variants/loss/nnUNetTrainerInstanceLosses.py): shared construction, validation, standard discovery, and run metadata.

Unlike upstream, the CC path is eager and makes no process-wide Dynamo configuration changes. Arithmetic and probabilities are float32 even inside AMP. The numerical reduction loops over samples while vectorizing over components; it does not construct a separate volume-sized prediction mask for every lesion. Network compilation still follows `nnUNet_compile`.

Training requires an NVIDIA CUDA GPU and matching CuPy. CPU/MPS, 2D configurations, binary datasets, overlapping regions, and ignore labels are rejected. Production has no silent CPU geometry fallback. Test-only SciPy geometry does not establish CUDA correctness.

The trainer name selects the loss. YAML `loss_type` may be absent/null, the parent's neutral legacy value `dice_ce`, or the matching selector (`cc_dice_ce` / `global_cc_dice_ce`). Other selections fail explicitly. Legacy `region_weights` are ignored with a warning in these class-based trainers. The existing parent behavior for missing YAML files is retained, including its default hyperparameters; use the explicit preset for reproducibility.

The shared [cc_multiclass.yaml](../../../configs/cc_multiclass.yaml) uses AdamW, LR `2e-4`, weight decay `3e-4`, 300 epochs, dropout `0.3`, null region weights, and no experiment suffix. These match the current root YAML's training settings; they are not tuned CC hyperparameters. Optimizer implementation, augmentation, patch sampling, architecture, and checkpoint selection remain inherited.

Each initialized run writes `cc_loss_config.json` containing objective, labels, weights, geometry/empty behavior, smoothing, upstream revision, versions, dataset/plan identity, and resolved training settings. The usual YAML backup and training log remain. Ordinary logs track the total objective; separate per-term epoch curves are not added by this integration.

## Install and run

From the `nnUNet_rain` repository root, in the intended training environment:

```bash
python -m pip install -e '.[dev]'
```

Install one CUDA-compatible CuPy distribution following the [official installation guide](https://docs.cupy.dev/en/stable/install.html). For example, a compatible CUDA 12 environment uses `cupy-cuda12x`; verify your runtime first. Do not install multiple CuPy distributions together. Do not install the Learning-to-Look-Closer fork beside this checkout: both provide `nnunetv2`.

Confirm the imported checkout and CLI come from the intended environment:

```bash
python -c "import nnunetv2; print(nnunetv2.__path__)"
which nnUNetv2_train
```

Set the usual `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results` locations. Then set this repository's shared preset and run either objective using the **same verified dataset, plan and fold**:

```bash
export NNUNET_CONFIG_PATH="$PWD/configs/cc_multiclass.yaml"
nnUNetv2_train DATASET_ID 3d_fullres FOLD -tr nnUNetTrainerCCDiceCE -p PLANS_NAME
nnUNetv2_train DATASET_ID 3d_fullres FOLD -tr nnUNetTrainerGlobalCCDiceCE -p PLANS_NAME
```

`DATASET_ID`, `FOLD`, and `PLANS_NAME` are placeholders, not runnable identifiers. Do not reuse the saved old Dataset001 WT/TC/ET preprocessing. Verify that the preprocessed dataset's metadata and actual masks consistently use the new labels. This implementation does not migrate existing data or checkpoints.

Standard `--c` resume works with the same trainer, plan, and training configuration. The normal result directories are distinct by trainer name. Keep the YAML/optimization settings unchanged when resuming; the inherited resume path does not restore optimizer schedule settings from the JSON sidecar automatically.

Prediction uses the standard interface, for example:

```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c 3d_fullres -f FOLD -tr nnUNetTrainerGlobalCCDiceCE -p PLANS_NAME
```

Alternatively use `nnUNetv2_predict_from_modelfolder -i INPUT_FOLDER -o OUTPUT_FOLDER -m MODEL_FOLDER -f FOLD`. This is useful if an experiment suffix changes the directory name. Prediction reconstructs the network from saved plans/checkpoint metadata and does not instantiate the training loss or read training YAML; CuPy is unnecessary for prediction.

## Validation and remaining gates

Run CPU mathematics and integration checks from this checkout:

```bash
python -m pytest nnunetv2/tests/test_cc_losses.py nnunetv2/tests/test_cc_trainers.py -q
```

The upstream reduction parity test reads only selected arithmetic functions from the pinned local checkout, without importing its global compiler settings. If the reference checkout is elsewhere, set `CC_UPSTREAM_ROOT` to its root. That test explicitly skips when the checkout is unavailable.

Local result: **31 passed**, using Python 3.13 / PyTorch 2.8.0 on Apple Silicon. Coverage includes adapters and gradients, original upstream reduction parity, explicit-loop parity, empty/full masks, false positives and missing lesions, core–edema gaps/merges/fragments, class averaging, global addition, deep supervision, invalid datasets/configs, CLI trainer discovery, metadata, checkpoint reload, and inference without CuPy/YAML. These are synthetic/CPU tests, not trained-model quality measurements.

Run the GPU acceptance suite on the training host:

```bash
python -m pytest nnunetv2/tests/test_cc_cuda.py -q -s
```

Local result: **13 skipped**, because CUDA is unavailable. On a CUDA host, missing/broken CuPy causes failures rather than silently skipping. Checks cover empty/full/touching masks, nearest-component validity including ties, non-default streams, AMP forward/backward, compiled/uncompiled networks, and two-GPU DDP gradient agreement with differing component counts. DDP tests skip if fewer than two GPUs are available. No GPU result is claimed here.

For loss-only timing and memory on a representative patch size:

```bash
CC_BENCH_SHAPE=160,192,160 python -m pytest nnunetv2/tests/test_cc_cuda.py -k profile -q -s
```

Choose dimensions from the actual plan. The profiler reports loss-forward/backward timing, PyTorch peak allocated memory, and CuPy pool reserved memory separately; PyTorch statistics do not include CuPy allocations. It is not a complete model-training benchmark. Before a long experiment, also smoke-test real training patches, all active decoder outputs, and full-step memory/time on the intended host. Set acceptable resource budgets there; none have been measured on this Mac.

## Interpretation limits

The matched CC term isolates the effect of adding global supervision. It does not prove that edema-component supervision improves tumor detection/counting, enforce core–edema attachment, or provide lesion IDs at inference. The output is a three-class semantic segmentation. Neither exact counts nor exact boundaries are guaranteed. Keep lesion definition, full-volume detection/counting evaluation, and any cascade separate from this implementation.
