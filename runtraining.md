# nnU-Net Pipeline Execution Guide (Server)

This guide provides the exact Docker commands required to execute the full nnU-Net pipeline on the lab virtual machine. All commands use the server's specific volume mount path:
`/mnt/local/data/rainsun/metastases/Rain-BrainMetastases-main/train/nnUNet_rain/nnUnet_dataset`

## 1. Verify Dataset Integrity
Before running the heavy preprocessing, it is best practice to quickly check that your dataset is formatted correctly and free of corruption. This extracts the fingerprint without doing the resampling.

```bash
docker run --gpus all -it --rm \
  -v /mnt/local/data/rainsun/metastases/Rain-BrainMetastases-main/train/nnUNet_rain/nnUnet_dataset:/workspace/nnunet_data \
  nnunet-rain "nnUNetv2_extract_fingerprint -d 1 --verify_dataset_integrity"
```

## 2. Planning and Preprocessing
Once integrity is verified, execute the heavy preprocessing. This will calculate the experiment plans, crop the images, standardize the spacing, and apply z-score normalization. The results are saved to `nnUNet_preprocessed/`.

```bash
docker run --gpus all -it --rm \
  -v /mnt/local/data/rainsun/metastases/Rain-BrainMetastases-main/train/nnUNet_rain/nnUnet_dataset:/workspace/nnunet_data \
  nnunet-rain "nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity"
```

## 3. Training

### Option A: Standard 3D Full-Resolution Baseline
To train the default single-stage `3d_fullres` model (e.g., using GPUs 0 and 1):
```bash
CUDA_VISIBLE_DEVICES=0,1 docker run --gpus all -it --rm --shm-size=32g \
  -v /mnt/local/data/rainsun/metastases/Rain-BrainMetastases-main/train/nnUNet_rain/nnUnet_dataset:/workspace/nnunet_data \
  nnunet-rain "nnUNetv2_train 1 3d_fullres 0 -num_gpus 2"
```

### Option B: Residual Encoder Configuration (Recommended for L40S/High VRAM)
If the default configuration plateaus, you can train a model using residual blocks instead of plain convolutional blocks. Note: The cascade configuration (`3d_lowres`) is automatically omitted by nnU-Net's planner if the dataset's cropped brain volume sizes are small enough to fit within a standard full-resolution patch. In such cases, the Residual Encoder (`ResEnc`) architecture is the primary method to improve feature representation.

#### Step 1: Generate the Residual Encoder Plan (within your container)
Run the experiment planner using the large preset:
```bash
docker run --gpus all -it --rm \
  -v /mnt/local/data/rainsun/metastases/Rain-BrainMetastases-main/train/nnUNet_rain/nnUnet_dataset:/workspace/nnunet_data \
  nnunet-rain "nnUNetv2_plan_experiment -d 1 -pl nnUNetPlannerResEncL"
```

#### Step 2: Train the Residual Encoder Model
Train the model by referencing the newly generated plans (`nnUNetResEncUNetLPlans`):
```bash
CUDA_VISIBLE_DEVICES=2 docker run --gpus all -it --rm --shm-size=32g \
  -v /mnt/local/data/rainsun/metastases/Rain-BrainMetastases-main/train/nnUNet_rain/nnUnet_dataset:/workspace/nnunet_data \
  nnunet-rain "nnUNetv2_train 1 3d_fullres 0 -p nnUNetResEncUNetLPlans -num_gpus 1"
```

### Option C: Custom Hyperparameter Tuning (Residual Encoder + Regularization)
For highly optimized training using custom spatial dropout, scaled batch size, AdamW optimizer, custom learning rate/weight decay, and early stopping at 500 epochs:

#### Step 1: Run Preprocessing for Custom Plans
Since the configuration (such as batch size and dropout) in the custom plans file `nnUNetResEncUNetLPlans_custom_0624.json` has been updated, you must run preprocessing for this specific plans file first:
```bash
docker run --gpus all -it --rm \
  -v /mnt/local/data/rainsun/metastases/Rain-BrainMetastases-main/train/nnUNet_rain/nnUnet_dataset:/workspace/nnunet_data \
  nnunet-rain "nnUNetv2_preprocess -d 1 -c 3d_fullres -plans nnUNetResEncUNetLPlans_custom_0624"
```

#### Step 2: Run Custom Training
Train the model using both the custom plans and the custom trainer class (`nnUNetTrainerMets`) on multi-GPU:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 docker run --gpus all -it --rm --shm-size=32g \
  -v /mnt/local/data/rainsun/metastases/Rain-BrainMetastases-main/train/nnUNet_rain/nnUnet_dataset:/workspace/nnunet_data \
  nnunet-rain "nnUNetv2_train 1 3d_fullres 0 -p nnUNetResEncUNetLPlans_custom_0624 -tr nnUNetTrainerMets -num_gpus 4"
```

## 4. Inference (Prediction)
After training is complete, you can generate predictions on new, unseen data. You will need to map an additional input directory containing the new raw scans, and an output directory for the predictions.

### Standard Full-Resolution Inference
```bash
docker run --gpus all -it --rm \
  -v /mnt/local/data/rainsun/metastases/Rain-BrainMetastases-main/train/nnUNet_rain/nnUnet_dataset:/workspace/nnunet_data \
  -v /path/to/server/input_scans:/workspace/input \
  -v /path/to/server/output_predictions:/workspace/output \
  nnunet-rain "nnUNetv2_predict -i /workspace/input -o /workspace/output -d 1 -c 3d_fullres -f 0"
```

### Residual Encoder Full-Resolution Inference
If you trained using the Residual Encoder plan, you must pass the matching plan name to the predictor:
```bash
docker run --gpus all -it --rm \
  -v /mnt/local/data/rainsun/metastases/Rain-BrainMetastases-main/train/nnUNet_rain/nnUnet_dataset:/workspace/nnunet_data \
  -v /path/to/server/input_scans:/workspace/input \
  -v /path/to/server/output_predictions:/workspace/output \
  nnunet-rain "nnUNetv2_predict -i /workspace/input -o /workspace/output -d 1 -c 3d_fullres -f 0 -p nnUNetResEncUNetLPlans"
```

### Custom Tuning Full-Resolution Inference
If you trained using the custom plans and custom trainer:
```bash
docker run --gpus all -it --rm \
  -v /mnt/local/data/rainsun/metastases/Rain-BrainMetastases-main/train/nnUNet_rain/nnUnet_dataset:/workspace/nnunet_data \
  -v /path/to/server/input_scans:/workspace/input \
  -v /path/to/server/output_predictions:/workspace/output \
  nnunet-rain "nnUNetv2_predict -i /workspace/input -o /workspace/output -d 1 -c 3d_fullres -f 0 -p nnUNetResEncUNetLPlans_custom_0624 -tr nnUNetTrainerMets"
```
*(Make sure to replace the `/path/to/server/...` paths with the actual paths to your test data)*