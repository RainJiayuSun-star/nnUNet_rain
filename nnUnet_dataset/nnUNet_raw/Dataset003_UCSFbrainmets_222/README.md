# Dataset003: UCSF 222-patient cohort

`ucsf_scan_A_preoperative_brats_training_222.csv` is the allowlist. It contains one A examination for each of 222 unique patients, with no recorded prior craniotomy/biopsy/resection and an available original BraTS segmentation. Patients with B/C or later examinations are retained through A only. Patient 100414 is excluded: A records prior surgery and has no BraTS segmentation; B is not substituted. This supersedes the overly restrictive 150-patient list.

This is the full cohort to partition into training/validation folds, not a predefined training split. Keep patients disjoint across all evaluation partitions, including UCSF cases renamed under BraTS. The source records do not independently date-verify the chronological first examination; describe the selection as A-indexed, not proven first-ever MRI or treatment-naive.

## Provenance

Verified on 2026-09-25 against `TableS1_UCSF_BrainMetastases_SubjectInfo.xlsx` (RENAMED, populated rows 2–462), the original 223-row `ucsf_first_preoperative_scans.csv`, and local original BraTS file availability. The spreadsheet SHA-256 is `b81910e1873cd133c4578c400657c2d52f3ae3c79f33922c812e64f260daba51`. The original 223-row CSV SHA-256 is `4acc6bd295dd1c1f7e148010998d1491f850a1d9a6af92e0311864a769e93aae`.

The supplied UCSF-BMSR paper, page 4, describes 324 preoperative examinations included in BraTS-METS 2023. These reduce to 223 patients; keeping eligible A examinations gives 222. Of these patients, 150 have one available examination and 72 have multiple. The last 99 formatted spreadsheet rows are blank, not auditable test-patient records.

CSV columns are portable: patient ID, examination ID, BraTS ID, A suffix, recorded surgical history, and total available examinations. `total_scans_in_source` is descriptive; values greater than one do not exclude a patient. There are no local workstation paths.

## Input and output format

The formatter selects by exact `subject_id` from the CSV and resolves files under the configured preprocessed source. It does not copy the original raw BraTS labels or perform registration, resampling or label conversion.

- FLAIR: `{subject}_FLAIR.nii.gz` or `{subject}_FLAIR_BC.nii.gz` → channel `0000`.
- T1post: `{subject}_T1post.nii.gz`, `_T1post_BC`, `_T1ce`, or `_T1ce_BC` → channel `0001`.
- Label: `{subject}_combined.seg.nii.gz` → `labelsTr`.

Suffix matching is case-insensitive. Exactly one candidate per channel/label must exist; both original and BC variants together are ambiguous and cause preflight to fail. Use a source directory containing only the intended processing stage. Combined labels retain the repository convention: background 0, tumor core 1, edema 2. The repository's `image_preprocessing/labeling/fastLabelCombine.py` creates these from BraTS labels. Their presence in the server's preprocessed directory must still be checked; original BraTS availability does not establish that the combined labels have been generated.

Sorted subject IDs map deterministically to `UCSFbrainmets_000` through `UCSFbrainmets_221`. A successful run produces 444 images, 222 labels, 222 mapping rows and `dataset.json` with `numTraining: 222`. The mapping retains `SubjectID,nnUNet_case_id,FLAIR,T1post,Label` and appends `patient_id,brats_id`.

## Run on the server

Use the official `eclipse-temurin:21-jdk` image for Java reformatting (no GPU needed). See [Adoptium container documentation](https://adoptium.net/installation/containers). Repository training instructions use `nnunet-rain`, but neither its Dockerfile nor the two preprocessing Dockerfiles explicitly install Java. Existing server containers have not been inspected.

The following mounts preserve both hard-coded Java defaults. Git records Dataset003 under `nnUNet_raw` (uppercase U), while this Mac also opens it through the user-provided `nnUnet_raw` spelling. The repository contains tracked paths using both spellings, which are distinct on Linux. The commands use the tracked Dataset003 spelling. Place all four delivered files together in that server directory. If your deployed server directory instead uses lowercase `nnUnet_raw`, change `RAW_HOST` below to that actual directory; the explicit mount preserves the Java container default either way.

```bash
SERVER_BASE=/mnt/local/data/rainsun/metastases
RAW_HOST="$SERVER_BASE/IDIA-BrainMetastases-main/train/nnUNet_rain/nnUnet_dataset/nnUNet_raw"
DATASET_HOST="$RAW_HOST/Dataset003_UCSFbrainmets_222"
SOURCE_HOST="$SERVER_BASE/datasets_preprocessed/UCSF_BrainMetastases_TRAIN"
DATASET_CONTAINER=/app/IDIA-BrainMetastases-main/train/nnUNet_rain/nnUnet_dataset/nnUNet_raw/Dataset003_UCSFbrainmets_222

# Preflight only; no generated files or output directories are written.
docker run --rm --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$SOURCE_HOST,dst=/app/datasets_preprocessed/UCSF_BrainMetastases_TRAIN,readonly" \
  --mount "type=bind,src=$DATASET_HOST,dst=$DATASET_CONTAINER" \
  --workdir "$DATASET_CONTAINER" \
  eclipse-temurin:21-jdk java reformat.java --dry-run

# Run only after preflight passes.
docker run --rm --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$SOURCE_HOST,dst=/app/datasets_preprocessed/UCSF_BrainMetastases_TRAIN,readonly" \
  --mount "type=bind,src=$DATASET_HOST,dst=$DATASET_CONTAINER" \
  --workdir "$DATASET_CONTAINER" \
  eclipse-temurin:21-jdk java reformat.java
```

Mount sources must already exist. Adjust the host variables to the actual server checkout if necessary. Only the source data are mounted read-only; the target needs write access for the executing UID/GID. Java source-file execution avoids leaving class files in the dataset.

Optional arguments (also supported after `javac reformat.java` using `java reformat`):

```text
java reformat.java [sourceRoot] [targetDatasetRoot] [cohortCsv] [--dry-run]
```

The default CSV is `ucsf_scan_A_preoperative_brats_training_222.csv` inside the resolved target. All 222 entries must be unique, valid A examinations with matching patient IDs, No surgical history and unique baseline BraTS IDs. The CSV is authoritative; Java does not reread the original Excel metadata. Never modify it casually to substitute cases.

Preflight reports missing, empty or ambiguous inputs and writes nothing on validation failure. Unlisted examinations are ignored. Existing outputs require the same mapping, including source paths, and byte-identical copied files. Unexpected files, symlinks, conflicting mappings, changed data or incompatible dataset metadata are rejected without deletion. Missing files from an interrupted run may be resumed when the mapping is intact. Use a separate clean target for a changed source or cohort. Rerun byte comparisons read all existing data and may take time. Do not run two formatters against the same target concurrently.

## nnU-Net integrity check and preprocessing

After a successful format, use the repository's `nnunet-rain` image with dataset ID **3**. The repository's `train/nnUNet_rain/readme.md` documents building it with `docker build -t nnunet-rain .` from that training directory. Mount the same `RAW_HOST` used for reformatting explicitly. The preprocessed and results directories use the repository's actual `nnUNet_preprocessed` and `nnUNet_results` capitalization.

```bash
NNUNET_DATA_HOST="$SERVER_BASE/IDIA-BrainMetastases-main/train/nnUNet_rain/nnUnet_dataset"
docker run --rm --shm-size=32g \
  --mount "type=bind,src=$NNUNET_DATA_HOST,dst=/workspace/nnunet_data" \
  -e nnUNet_raw=/workspace/nnunet_data/nnUnet_raw \
  -e nnUNet_preprocessed=/workspace/nnunet_data/nnUnet_preprocessed \
  -e nnUNet_results=/workspace/nnunet_data/nnUNet_results \
  nnunet-rain "nnUNetv2_plan_and_preprocess -d 3 --verify_dataset_integrity"
```

The integrity check examines actual NIfTI structure, image/label geometry and label values; Java only validates selection, file availability and copy consistency. Do not reuse Dataset001's preprocessed data, splits or trained model as if it were Dataset003. Subsequent training uses `nnunet-rain`, dataset 3 and the appropriate GPU configuration from `runtraining.md`.

## Local regression checks

Run `python3 test_reformat.py` with Java 21 (`java` and `javac`) on PATH. Tests create temporary synthetic byte fixtures, not clinical images. They exercise cohort selection, validation, naming, byte-preserving copies and rerun protection. NIfTI validity must be checked separately on the server as above. No server conversion, preprocessing or training was performed during implementation.
