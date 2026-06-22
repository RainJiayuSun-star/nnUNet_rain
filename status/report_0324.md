# Report: 3.10-3.24 Progress

## Current Status on nnUNet Model Training

- nnU-Net dataset conversion for UCSF brain metastases is completed under `nnUnet_dataset/nnUnet_raw/Dataset001_UCSFbrainmets`.
- Current training setup is **4-channel**:
  - `0000`: FLAIR
  - `0001`: T1pre
  - `0002`: T1post
  - `0003`: T2Synth
- `dataset.json` currently reports:
  - `numTraining = 324`
  - labels: background + 3 foreground classes
- Case traceability from original IDs to renamed nnU-Net IDs is available in:
  - `nnUnet_dataset/nnUnet_raw/Dataset001_UCSFbrainmets/renaming_map.csv`

### Data Engineering Work Completed

- Implemented `reformat.java` to:
  - scan source NIfTI files
  - enforce complete-case requirement (all required modalities + BraTS segmentation)
  - skip missing-label/missing-modality cases
  - copy and rename into nnU-Net format (`imagesTr`, `labelsTr`)
  - generate `renaming_map.csv`
  - generate/update `dataset.json`
  - print per-case progress logs and skip notifications
- Implemented `reformat_clean.java` to:
  - remove `*_0003.nii.gz` channel files (T2/T2Synth) when running 3-channel experiments
  - rewrite `dataset.json` for 3-channel configuration
  - report deletion counts and consistency warnings

### Preprocessing / Planning Result

- Executed:
  - `nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity`
- Outcome:
  - dataset integrity check passed
  - fingerprint extraction completed for 324 cases
  - plans generated successfully
  - preprocessing completed for `2d` and `3d_fullres` (324/324)
  - `3d_lowres` was automatically skipped (expected; size difference too small)

### Logged Run Details (Captured)

- Reader/writer selected: `SimpleITKIO`
- Integrity check message: **Done** (no errors reported)
- Fingerprint extraction:
  - `324/324` cases
  - runtime shown: `00:57`
  - throughput shown: `5.65 it/s`
- Planner note:
  - using nnU-Net default planner with recommendation notice for newer ResEnc presets
  - reference: `documentation/resenc_presets.md` in upstream nnU-Net repo
- Automatic configuration decision:
  - dropped `3d_lowres` because size difference vs `3d_fullres` was too small
  - reported median size comparison:
    - `3d_fullres`: `[100.5, 227, 215]`
    - `3d_lowres`: `[100, 227, 215]`
- Planned 2D config (key values):
  - `batch_size: 56`
  - `patch_size: [256, 224]`
  - `spacing: [0.8594, 0.8594]`
  - normalization: 4x `ZScoreNormalization`
- Planned 3D fullres config (key values):
  - `batch_size: 2`
  - `patch_size: [80, 192, 160]`
  - `spacing: [1.5, 0.8594, 0.8594]`
  - normalization: 4x `ZScoreNormalization`
- Plans saved to:
  - `nnUnet_dataset/nnUnet_preprocessed/Dataset001_UCSFbrainmets/nnUNetPlans.json`
- Preprocessing runtimes shown:
  - `2d`: `324/324` in `19:15` (`3.57 s/it`)
  - `3d_fullres`: `324/324` in `25:37` (`4.75 s/it`)
  - `3d_lowres`: skipped (not present in plans)
```
(nnUnet) rainsun@DESKTOP-9VE70J6:/mnt/d/A1_RainSun_20240916/1-UWMadison/IDiA-Lab/brain_metastases_train/nnUnet_dataset/nnUnet_raw/Dataset001_UCSFbrainmets$ nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
Fingerprint extraction...
Dataset001_UCSFbrainmets
Using <class 'nnunetv2.imageio.simpleitk_reader_writer.SimpleITKIO'> as reader/writer

####################
verify_dataset_integrity Done. 
If you didn't see any error messages then your dataset is most likely OK!
####################

Using <class 'nnunetv2.imageio.simpleitk_reader_writer.SimpleITKIO'> as reader/writer
Extracting dataset fingerprint: 100%|███████████████████████████████████████| 324/324 [00:57<00:00,  5.65it/s]
Experiment planning...

############################
INFO: You are using the old nnU-Net default planner. We have updated our recommendations. Please consider using those instead! Read more here: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md
############################

Dropping 3d_lowres config because the image size difference to 3d_fullres is too small. 3d_fullres: [100.5 227.  215. ], 3d_lowres: [100, 227, 215]
2D U-Net configuration:
{'data_identifier': 'nnUNetPlans_2d', 'preprocessor_name': 'DefaultPreprocessor', 'batch_size': 56, 'patch_size': (np.int64(256), np.int64(224)), 'median_image_size_in_voxels': array([227., 215.]), 'spacing': array([0.85939997, 0.85939997]), 'normalization_schemes': ['ZScoreNormalization', 'ZScoreNormalization', 'ZScoreNormalization', 'ZScoreNormalization'], 'use_mask_for_norm': [True, True, True, True], 'resampling_fn_data': 'resample_data_or_seg_to_shape', 'resampling_fn_seg': 'resample_data_or_seg_to_shape', 'resampling_fn_data_kwargs': {'is_seg': False, 'order': 3, 'order_z': 0, 'force_separate_z': None}, 'resampling_fn_seg_kwargs': {'is_seg': True, 'order': 1, 'order_z': 0, 'force_separate_z': None}, 'resampling_fn_probabilities': 'resample_data_or_seg_to_shape', 'resampling_fn_probabilities_kwargs': {'is_seg': False, 'order': 1, 'order_z': 0, 'force_separate_z': None}, 'architecture': {'network_class_name': 'dynamic_network_architectures.architectures.unet.PlainConvUNet', 'arch_kwargs': {'n_stages': 6, 'features_per_stage': (32, 64, 128, 256, 512, 512), 'conv_op': 'torch.nn.modules.conv.Conv2d', 'kernel_sizes': ((3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3)), 'strides': ((1, 1), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)), 'n_conv_per_stage': (2, 2, 2, 2, 2, 2), 'n_conv_per_stage_decoder': (2, 2, 2, 2, 2), 'conv_bias': True, 'norm_op': 'torch.nn.modules.instancenorm.InstanceNorm2d', 'norm_op_kwargs': {'eps': 1e-05, 'affine': True}, 'dropout_op': None, 'dropout_op_kwargs': None, 'nonlin': 'torch.nn.LeakyReLU', 'nonlin_kwargs': {'inplace': True}}, '_kw_requires_import': ('conv_op', 'norm_op', 'dropout_op', 'nonlin')}, 'batch_dice': True}

Using <class 'nnunetv2.imageio.simpleitk_reader_writer.SimpleITKIO'> as reader/writer
3D fullres U-Net configuration:
{'data_identifier': 'nnUNetPlans_3d_fullres', 'preprocessor_name': 'DefaultPreprocessor', 'batch_size': 2, 'patch_size': (np.int64(80), np.int64(192), np.int64(160)), 'median_image_size_in_voxels': array([100.5, 227. , 215. ]), 'spacing': array([1.5       , 0.85939997, 0.85939997]), 'normalization_schemes': ['ZScoreNormalization', 'ZScoreNormalization', 'ZScoreNormalization', 'ZScoreNormalization'], 'use_mask_for_norm': [True, True, True, True], 'resampling_fn_data': 'resample_data_or_seg_to_shape', 'resampling_fn_seg': 'resample_data_or_seg_to_shape', 'resampling_fn_data_kwargs': {'is_seg': False, 'order': 3, 'order_z': 0, 'force_separate_z': None}, 'resampling_fn_seg_kwargs': {'is_seg': True, 'order': 1, 'order_z': 0, 'force_separate_z': None}, 'resampling_fn_probabilities': 'resample_data_or_seg_to_shape', 'resampling_fn_probabilities_kwargs': {'is_seg': False, 'order': 1, 'order_z': 0, 'force_separate_z': None}, 'architecture': {'network_class_name': 'dynamic_network_architectures.architectures.unet.PlainConvUNet', 'arch_kwargs': {'n_stages': 6, 'features_per_stage': (32, 64, 128, 256, 320, 320), 'conv_op': 'torch.nn.modules.conv.Conv3d', 'kernel_sizes': ((3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)), 'strides': ((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (1, 2, 2)), 'n_conv_per_stage': (2, 2, 2, 2, 2, 2), 'n_conv_per_stage_decoder': (2, 2, 2, 2, 2), 'conv_bias': True, 'norm_op': 'torch.nn.modules.instancenorm.InstanceNorm3d', 'norm_op_kwargs': {'eps': 1e-05, 'affine': True}, 'dropout_op': None, 'dropout_op_kwargs': None, 'nonlin': 'torch.nn.LeakyReLU', 'nonlin_kwargs': {'inplace': True}}, '_kw_requires_import': ('conv_op', 'norm_op', 'dropout_op', 'nonlin')}, 'batch_dice': False}

Plans were saved to /mnt/d/A1_RainSun_20240916/1-UWMadison/IDiA-Lab/brain_metastases_train/nnUnet_dataset/nnUnet_preprocessed/Dataset001_UCSFbrainmets/nnUNetPlans.json
Preprocessing...
Preprocessing dataset Dataset001_UCSFbrainmets
Configuration: 2d...
{'data_identifier': 'nnUNetPlans_2d', 'preprocessor_name': 'DefaultPreprocessor', 'batch_size': 56, 'patch_size': [256, 224], 'median_image_size_in_voxels': [227.0, 215.0], 'spacing': [0.8593999743461609, 0.8593999743461609], 'normalization_schemes': ['ZScoreNormalization', 'ZScoreNormalization', 'ZScoreNormalization', 'ZScoreNormalization'], 'use_mask_for_norm': [True, True, True, True], 'resampling_fn_data': 'resample_data_or_seg_to_shape', 'resampling_fn_seg': 'resample_data_or_seg_to_shape', 'resampling_fn_data_kwargs': {'is_seg': False, 'order': 3, 'order_z': 0, 'force_separate_z': None}, 'resampling_fn_seg_kwargs': {'is_seg': True, 'order': 1, 'order_z': 0, 'force_separate_z': None}, 'resampling_fn_probabilities': 'resample_data_or_seg_to_shape', 'resampling_fn_probabilities_kwargs': {'is_seg': False, 'order': 1, 'order_z': 0, 'force_separate_z': None}, 'architecture': {'network_class_name': 'dynamic_network_architectures.architectures.unet.PlainConvUNet', 'arch_kwargs': {'n_stages': 6, 'features_per_stage': [32, 64, 128, 256, 512, 512], 'conv_op': 'torch.nn.modules.conv.Conv2d', 'kernel_sizes': [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]], 'strides': [[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]], 'n_conv_per_stage': [2, 2, 2, 2, 2, 2], 'n_conv_per_stage_decoder': [2, 2, 2, 2, 2], 'conv_bias': True, 'norm_op': 'torch.nn.modules.instancenorm.InstanceNorm2d', 'norm_op_kwargs': {'eps': 1e-05, 'affine': True}, 'dropout_op': None, 'dropout_op_kwargs': None, 'nonlin': 'torch.nn.LeakyReLU', 'nonlin_kwargs': {'inplace': True}}, '_kw_requires_import': ['conv_op', 'norm_op', 'dropout_op', 'nonlin']}, 'batch_dice': True}
Preprocessing cases: 100%|██████████████████████████████████████████████████| 324/324 [19:15<00:00,  3.57s/it]
Configuration: 3d_fullres...
{'data_identifier': 'nnUNetPlans_3d_fullres', 'preprocessor_name': 'DefaultPreprocessor', 'batch_size': 2, 'patch_size': [80, 192, 160], 'median_image_size_in_voxels': [100.5, 227.0, 215.0], 'spacing': [1.5, 0.8593999743461609, 0.8593999743461609], 'normalization_schemes': ['ZScoreNormalization', 'ZScoreNormalization', 'ZScoreNormalization', 'ZScoreNormalization'], 'use_mask_for_norm': [True, True, True, True], 'resampling_fn_data': 'resample_data_or_seg_to_shape', 'resampling_fn_seg': 'resample_data_or_seg_to_shape', 'resampling_fn_data_kwargs': {'is_seg': False, 'order': 3, 'order_z': 0, 'force_separate_z': None}, 'resampling_fn_seg_kwargs': {'is_seg': True, 'order': 1, 'order_z': 0, 'force_separate_z': None}, 'resampling_fn_probabilities': 'resample_data_or_seg_to_shape', 'resampling_fn_probabilities_kwargs': {'is_seg': False, 'order': 1, 'order_z': 0, 'force_separate_z': None}, 'architecture': {'network_class_name': 'dynamic_network_architectures.architectures.unet.PlainConvUNet', 'arch_kwargs': {'n_stages': 6, 'features_per_stage': [32, 64, 128, 256, 320, 320], 'conv_op': 'torch.nn.modules.conv.Conv3d', 'kernel_sizes': [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]], 'strides': [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]], 'n_conv_per_stage': [2, 2, 2, 2, 2, 2], 'n_conv_per_stage_decoder': [2, 2, 2, 2, 2], 'conv_bias': True, 'norm_op': 'torch.nn.modules.instancenorm.InstanceNorm3d', 'norm_op_kwargs': {'eps': 1e-05, 'affine': True}, 'dropout_op': None, 'dropout_op_kwargs': None, 'nonlin': 'torch.nn.LeakyReLU', 'nonlin_kwargs': {'inplace': True}}, '_kw_requires_import': ['conv_op', 'norm_op', 'dropout_op', 'nonlin']}, 'batch_dice': False}
Preprocessing cases: 100%|██████████████████████████████████████████████████| 324/324 [25:37<00:00,  4.75s/it]
Configuration: 3d_lowres...
INFO: Configuration 3d_lowres not found in plans file nnUNetPlans.json of dataset Dataset001_UCSFbrainmets. Skipping.
```
### Containerization Work

- Added GPU-ready Docker setup in:
  - `nnUNet_rain/Dockerfile`
- Docker image is configured to:
  - use CUDA-enabled PyTorch runtime
  - install nnU-Net from the local fork
  - expose `nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results` via environment variables
  - support multi-GPU execution via `docker run --gpus ...`

### Next Steps

1. Train fold 0 baseline (4 channels):
   - `nnUNetv2_train 1 3d_fullres 0`
2. Launch folds 1-4 (serially or in parallel on available GPUs).
3. Collect baseline metrics from validation.
4. Run 3-channel ablation (drop `_0003`) using `reformat_clean.java`.
5. Re-run planning/preprocessing if dataset channel structure changes.
6. Compare 4-channel vs 3-channel performance before final model selection.

### Notes

- The 4-channel baseline is intentionally preserved first to quantify whether `T2Synth` helps or hurts.
- If training is moved to the shared lab VM, Docker is the preferred reproducible path to avoid conda conflicts.
---

## Current Datasets Found

Cancer Imaging archive: links to current open-access brain/head-neck datasets and metastases collections: [https://www.cancerimagingarchive.net/brain-and-head-neck-imaging-still-available-on-tcia/](https://www.cancerimagingarchive.net/brain-and-head-neck-imaging-still-available-on-tcia/)

### PKG - Pretreat-MetsToBrain-Masks
- Link: [https://www.cancerimagingarchive.net/collection/pretreat-metstobrain-masks/](https://www.cancerimagingarchive.net/collection/pretreat-metstobrain-masks/)
- Link to paper: [https://pubmed.ncbi.nlm.nih.gov/38424079/](https://pubmed.ncbi.nlm.nih.gov/38424079/)
- Note: Expert manual segmentations provided; pre-treatment cohort
- **Number of Subjects: 200**
- Source: Yale New Haven Health database (2013-2021), Yale tumor board registry (2021), Yale gamma knife registry (2017-2021)
- Modalities/Sequences: T1w, T1 post-gadolinium, T2w, FLAIR + expert masks
- Image preprocessing: Sequences exported to NIfTI, co-registered to SRI24 template, resampled to 1 mm isotropic, skull stripped; segmentation masks registered and quality checked
- Image acquisition/parameters: 1.5T/3T scanners, mainly MPRAGE for post-contrast T1, subset with spin-echo
- Segmentation protocol: Manual segmentations in research PACS, approved by neuroradiologists
- Registration approach: **Common-template registration** (explicit SRI24 atlas registration for all cases)

### Brain-Mets-Lung-MRI-Path-Segs
MR imaging and segmentations with matched brain biopsy pathology slides from patients with lung-primary brain metastases.
- Link: [https://www.cancerimagingarchive.net/collection/brain-mets-lung-mri-path-segs/](https://www.cancerimagingarchive.net/collection/brain-mets-lung-mri-path-segs/)
- Number of Subjects: 103
- Note: Includes imaging + segmentations + matched pathology context (radiology-pathology correlation use cases)
- Registration approach: **Likely per-case matching** (T1CE/FLAIR matched with segmentations); no explicit statement of one global atlas warp in collection summary

### PKG - Yale-Brain-Mets-Longitudinal
- Link: [https://www.cancerimagingarchive.net/collection/yale-brain-mets-longitudinal/](https://www.cancerimagingarchive.net/collection/yale-brain-mets-longitudinal/)
- DOI: `10.7937/3YAT-E768`
- Snapshot:
  - Subjects: **1,430**
  - Studies: **11,892** longitudinal MRI studies
  - Size: **43 GB**
  - MRI sequences: T1 pre-contrast, T1 post-contrast, T2, FLAIR
  - Includes associated clinical/scanner metadata spreadsheet
- Note: Longitudinal cohort; segmentation masks are not emphasized as a primary deliverable in the collection summary.
- Registration approach: **Likely native/per-study space** (standardized sequence selection + brain extraction; no explicit common-template registration noted)

### UCSF_BrainMetastases_v1.3
- Link: [https://imagingdatasets.ucsf.edu/dataset/1](https://imagingdatasets.ucsf.edu/dataset/1)
- Number of Subjects: 412
- Notes:
  - Public release includes registered/skull-stripped multimodal MRI in NIfTI
  - Current nnU-Net training set prepared in this project: 324 complete labeled cases
- Registration approach: **Mixed by release context** (general release is registered/skull-stripped; BraTS-related notes indicate availability in both subject-native and BraTS SRI atlas spaces)

### Stanford (BrainMetShare)
- Link: [https://aimi.stanford.edu/datasets/brainmetshare](https://aimi.stanford.edu/datasets/brainmetshare)
- Number of Studies: 156 whole-brain MRI studies
- Dataset DOI: [https://doi.org/10.71718/z66c-qr59](https://doi.org/10.71718/z66c-qr59)
- Notes:
  - Multi-sequence pre/post-contrast MRI, co-registered and skull-stripped
  - Provider-described split:
    - 105 cases with radiologist segmentation labels
    - 51 unlabeled test cases

---

## Inference & Benchmark 

### BraTS Challenge
- Link: https://github.com/BrainLesion/BraTS
- Provides good baseline to compare against

### -> TODO: Set up Inference code for current model weights/checkpoints we have