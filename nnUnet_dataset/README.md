# nnU-Net Dataset Structure & Data Preparation

This directory contains the essential structure for training models with nnU-Net, alongside the data engineering scripts used to reformat raw datasets into nnU-Net's strict structure.

## Directory Structure
The expected structure for nnU-Net is as follows:

```
nnUnet_dataset/
├── nnUnet_raw/
│   └── Dataset001_UCSFbrainmets/
│       ├── imagesTr/         # Contains training images (*_0000.nii.gz, *_0001.nii.gz...)
│       ├── labelsTr/         # Contains training labels (no modality suffix)
│       ├── dataset.json      # Metadata file describing channels and classes
│       ├── reformat.java     # Script to generate nnU-Net raw dataset layout
│       └── reformat_clean.java # Script for dataset ablation (e.g., removing a channel)
├── nnUnet_preprocessed/      # Populated automatically by nnUNetv2_plan_and_preprocess
└── nnUnet_trained_models/    # Populated automatically by nnUNetv2_train
```

## Data Formatting Scripts
We provide two Java scripts to assist in translating generic or custom datasets into the specific format nnU-Net requires.

### `reformat.java`
**Purpose**: Transforms raw NIfTI files into the standard nnU-Net `imagesTr` and `labelsTr` layout.
- Filters out incomplete cases.
- Assigns contiguous sequential case IDs (e.g., `UCSFbrainmets_000`).
- Appends the correct channel ID suffixes (e.g., `_0000` for FLAIR, `_0001` for T1pre, etc.).
- Generates `renaming_map.csv` for case traceability.
- Generates the required `dataset.json` metadata.

### `reformat_clean.java`
**Purpose**: Ablation tool to adjust existing nnU-Net datasets (e.g., converting a 4-channel dataset to a 3-channel dataset).
- Deletes specific channel files (like `_0003.nii.gz`) from `imagesTr`.
- Automatically rewrites `dataset.json` to reflect the updated channel count, which is required before re-running `nnUNetv2_plan_and_preprocess`.

## How to use for a new dataset
1. Place your newly preprocessed/bias-corrected files into a new `DatasetXXX_Name` directory within `nnUnet_raw/`.
2. Format them either manually or using an updated `reformat.java` script to split images into `imagesTr/` with 4-digit channel suffixes, and labels into `labelsTr/`.
3. Provide the `dataset.json`.
4. Ensure `nnUnet_preprocessed/` and `nnUnet_trained_models/` remain empty. 
5. Run the standard nnU-Net preprocessing and training pipelines; nnU-Net will handle populating the rest.
