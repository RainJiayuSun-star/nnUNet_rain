# Binary whole-metastasis CC-DiceCE

## Current contract

Foreground is enhancing plus non-enhancing/necrotic metastasis, **excluding edema**.
The network learns the combined binary mask directly. Edema is supervised as
background, not ignored. MRI inputs remain FLAIR and T1post.

This supersedes the three-class equal core/edema loss and AdamW YAML recipe.
Reference: TIO-IKIM/Learning-to-Look-Closer commit
`e5013bdb560eee544a94e1f068cb5cb4d003a51c`, Apache-2.0.

- `nnUNetTrainerDiceCEBaseline`: global binary Dice+CE, smooth=0.
- `nnUNetTrainerCCDiceCE`: binary instance-wise CC Dice+CE only.
- `nnUNetTrainerGlobalCCDiceCE`: global plus CC Dice+CE, coefficients 1:1; primary candidate.

All inherit ordinary nnUNetTrainer directly with a shared binary-loss mixin: SGD, LR 0.01,
weight decay 3e-5, momentum 0.99, Nesterov, polynomial decay (exponent 0.9),
1,000 epochs, 250 training/50 validation iterations per epoch, standard
augmentation, and plan-derived architecture/sampling. They never read YAML or
change planned dropout. DA5 and legacy configurable trainers are unchanged.
Standard nnUNetTrainer remains unchanged with smooth=1e-5.
Resolved settings are saved to each fold's `cc_loss_config.json`.

## Mathematics and adaptations

Compute 26-connected GT foreground components and Euclidean voxel-distance
Voronoi regions. For each region R:

```text
CC_R = 1 - 2 sum_R(p*y) / max(sum_R(p)+sum_R(y), 1e-7)
       + mean_R(binary CE)
L_CC = mean_samples(mean_components(CC_R))
L_GlobalCC = L_global + L_CC
```

CC CE uses softmax probabilities clamped at 1e-7, matching the reference.
Empty foreground returns mean binary CE + 1; false positives still have gradients.
Global Dice is negative Dice plus CE, excludes background from Dice, includes
both classes in CE, and clamps its denominator at 1e-8. Neither adds smoothing.

Apply the full objective at every active deep-supervision scale, with normalized
1/2^s weights and recomputed geometry. The last scale is zero except the native
noncompiled-DDP safeguard of 1e-6; a single output has weight 1.
DDP uses local sample means followed by rank averaging, as upstream does;
use equal local batch sizes for a sample-matched multi-GPU comparison.

Production geometry requires CUDA/CuPy; the baseline and inference do not.
Differences from the reference: lazy imports, explicit current device/stream,
empty-mask guard, FP32 softmax/arithmetic under AMP, and eager sample/scatter
reduction rather than compiled/vmapped reduction. Network compilation remains
native. CPU SciPy geometry is a test oracle only. No upstream global Dynamo
settings or Blob implementations are imported.

## Binary dataset migration

The converter accepts **combined** labels with metadata background=0,
tumor_core=1, edema=2. It maps 1 to whole_metastasis=1 and 0/2 to background=0.
It rejects unexpected IDs, mismatched geometry, missing image channels,
incomplete patient mappings, and conflicting outputs before writing.

Use a distinct new dataset ID; Dataset004 below is an example, not a reserved
server ID. Verify it is unused in raw, preprocessed and results roots.
Images are copied unchanged; NIfTI label geometry, case IDs, channels and patient
mapping are preserved. The manifest records input/output label checksums,
image checksums, voxel counts, source metadata/mapping hashes and audit status.
Existing destinations are refused, including partial conversions. A completed
conversion has `conversion_manifest.json`.

The source label-combining script intends to merge raw IDs 1/3/4 into label 1.
Verify actual annotation provenance. Optional `--audit-dir` takes same-space raw
annotations named CASE_ID.nii.gz, or the original UCSF layout
SubjectID/SubjectID_BraTS-seg.nii.gz (resolved using renaming_map.csv), and checks
that union against combined label 1. Standard *_seg files are not guessed as a fallback.
Different geometry, unknown IDs or mismatched masks fail; absent annotations
are explicitly reported as not verified. The converter does not register or
resample. Review unaudited cases before clinical experiments.

## Server commands

Use the existing CUDA 12 base image and a derived CuPy image. Change the host
raw path if its spelling is nnUnet_raw rather than nnUNet_raw; Linux is
case-sensitive. The explicit environment overrides keep container paths aligned.

```bash
PROJECT=/mnt/local/data/rainsun/metastases/IDIA-BrainMetastases-main/train/nnUNet_rain
DATA="$PROJECT/nnUnet_dataset"
RAW="$DATA/nnUNet_raw"

docker build -t nnunet-rain-cc - <<'EOF'
FROM nnunet-rain
RUN python -m pip install cupy-cuda12x pytest
EOF

cc_run() {
  docker run --runtime=nvidia --rm --shm-size=32g \
    -e NVIDIA_VISIBLE_DEVICES=3 -e CUDA_VISIBLE_DEVICES=0 \
    --env-file "$PROJECT/.env" \
    -e nnUNet_raw=/workspace/raw \
    -e nnUNet_preprocessed=/workspace/nnunet_data/nnUNet_preprocessed \
    -e nnUNet_results=/workspace/nnunet_data/nnUNet_results \
    -v "$PROJECT:/opt/nnunet" -v "$DATA:/workspace/nnunet_data" \
    -v "$RAW:/workspace/raw" -w /opt/nnunet nnunet-rain-cc "$1"
}

cc_run 'python -m nnunetv2.dataset_conversion.convert_whole_metastasis \
  --source /workspace/raw/Dataset003_UCSFbrainmets_222 \
  --destination /workspace/raw/Dataset004_UCSFWholeMetastases_222 \
  --splits /workspace/nnunet_data/nnUNet_preprocessed/Dataset003_UCSFbrainmets_222/splits_final.json \
  --dry-run'
```

Repeat without --dry-run after review. Supply --audit-dir when originals are
available, mounting that directory into the container. If source folds do not
exist, omit --splits; use the single generated binary split file across all three
trainers. Multiple scans per patient require supplied patient-disjoint splits.
Supplied folds must cover exactly the same cases with each case in validation
exactly once. Patient provenance is copied as renaming_map.csv.

```bash
cc_run 'nnUNetv2_plan_and_preprocess -d 4 -c 3d_fullres --verify_dataset_integrity'

# Only if --splits was supplied. Refuse to overwrite differing existing folds.
cc_run 'cp -n /workspace/raw/Dataset004_UCSFWholeMetastases_222/splits_final.json \
  /workspace/nnunet_data/nnUNet_preprocessed/Dataset004_UCSFWholeMetastases_222/splits_final.json && \
  cmp /workspace/raw/Dataset004_UCSFWholeMetastases_222/splits_final.json \
  /workspace/nnunet_data/nnUNet_preprocessed/Dataset004_UCSFWholeMetastases_222/splits_final.json'

cc_run 'python -m pytest nnunetv2/tests/test_cc_losses.py \
  nnunetv2/tests/test_cc_trainers.py nnunetv2/tests/test_whole_metastasis_conversion.py \
  nnunetv2/tests/test_cc_cuda.py -q -rs'
```

Set CC_UPSTREAM_ROOT to the mounted pinned reference checkout to enable the
upstream arithmetic parity test in the container. It loads selected functions
by AST, not a second nnunetv2 package. Without CUDA the GPU suite skips;
on one GPU only the two DDP cases skip. Missing CuPy on CUDA is a failure.

## Real-data smoke test and training

Smoke tests use native initialization and actual preprocessed/augmented batches,
validation, and checkpoint reload. A new isolated output directory is required
for every invocation. They disable W&B and do not replace production checkpoints.

```bash
for trainer in nnUNetTrainerDiceCEBaseline nnUNetTrainerCCDiceCE nnUNetTrainerGlobalCCDiceCE; do
  cc_run "python -m nnunetv2.run.cc_training_smoke 4 --trainer $trainer \
    --plans nnUNetPlans --fold 0 --steps 3 \
    --output-dir /workspace/nnunet_data/cc_smoke/$trainer" || break
done
```

Monitor physical GPU 3 separately using `nvidia-smi -i 3 -l 1`. The smoke report
records step time, PyTorch peak allocated/reserved memory and CuPy retained pool
memory. These exclude some CUDA/context allocations and are not a synchronized
combined memory peak. There is no measured 3-5 GiB overhead guarantee.

Only after all gates pass, run fresh binary fold-0 experiments:

```bash
for trainer in nnUNetTrainerDiceCEBaseline nnUNetTrainerCCDiceCE nnUNetTrainerGlobalCCDiceCE; do
  cc_run "nnUNetv2_train 4 3d_fullres 0 -p nnUNetPlans -tr $trainer -num_gpus 1" || break
done
```

Inspect fold-0 results before repeating folds 1-4. Keep the same plans, splits,
training budget and checkpoint selection. Each trainer uses its own native result
directory. Resume binary training with --c and identical dataset/plan/trainer.
Never resume old three-class AdamW checkpoints or reuse multiclass preprocessing.
Prediction requires the trainer source but neither CuPy nor YAML:

```bash
cc_run 'nnUNetv2_predict -i /workspace/nnunet_data/inference_input \
  -o /workspace/nnunet_data/inference_output_globalcc -d 4 -c 3d_fullres \
  -p nnUNetPlans -tr nnUNetTrainerGlobalCCDiceCE -f 0'
```

## Evaluation and limits

Report whole-metastasis Dice, lesion recall/precision/F1, false positives per
scan and per-scan count error, including lesion-size strata and splits/merges.
Keep 26-connectivity and one-to-one matching fixed across runs; reference-style
detection requires at least one overlapping voxel. Fix assignment/tie-breaking
before comparisons and do not tune thresholds on held-out test cases.
Do not remove all but the largest component in a multi-metastasis task.
CC is not a direct count loss: touching lesions may merge and fragmented
annotations may split. The integration supplies training tools; no server
conversion, CUDA acceptance or long training is implied by CPU tests.

## Verification performed locally (2026-10-21)

An isolated Python 3.11 environment with PyTorch 2.14.1 ran:

```bash
python -m pytest nnunetv2/tests/test_cc_losses.py \
  nnunetv2/tests/test_cc_trainers.py nnunetv2/tests/test_cc_cuda.py \
  nnunetv2/tests/test_whole_metastasis_conversion.py -q -rs
```

Result: **44 passed, 19 skipped**, with one dependency deprecation warning.
Seventeen skipped cases require CUDA and two require two-GPU DDP. These are
unverified, not passed. The server image/library versions must run acceptance
independently. Targeted Ruff checks and `git diff --check` passed.

Passing CPU checks include explicit and pinned-upstream value/gradient parity,
empty/full/unequal lesions and false positives, coarse-scale disappearance,
FP32 global arithmetic, actual native initialization with null dropout kwargs,
SGD settings, train/validation/checkpoint plumbing, two-channel inference without
CuPy, and converter integrity/split/conflict tests. CC trainer plumbing on CPU
uses a clearly marked SciPy geometry stub; production has no CPU geometry fallback.
The real-data smoke CLI imports and displays help; its CUDA execution remains pending.

A read-only converter dry run against the local
`2026Fall/Dataset003_UCSFbrainmets_222`, using `2026Fall/splits_final.json`
and `Datasets/UCSF_BrainMetastases_v1.3/UCSF_BrainMetastases_TRAIN`
as the audit directory, passed all **222 cases**:

- Expected label values, both input channels, image/label geometry, case identities,
  and patient-disjoint fold assignments validated.
- All 222 combined label-1 masks exactly matched the union of raw annotation IDs
  1/3/4 in matching image space; zero unaudited cases.
- No binary dataset was written and Dataset003 was not modified.

This verifies consistency with the available source annotations, not independent
clinical correctness or the contents of a different server copy. Repeat the dry
run and provenance audit on the server before conversion. CUDA geometry/AMP,
network compilation, DDP, real-data train/validation/resume, and measured VRAM
remain explicit acceptance gates.
