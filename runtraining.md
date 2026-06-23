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
Train the model using the preprocessed data. The following command uses the 3D full-resolution configuration on Fold 0.
*(To utilize the 4x L40 GPUs, we add `-num_gpus 4` and use the heavy Residual Encoder architecture `-tr nnUNetTrainerResencL`)*

```bash
CUDA_VISIBLE_DEVICES=0,1 docker run --gpus all -it --rm --shm-size=32g \
  -v /mnt/local/data/rainsun/metastases/Rain-BrainMetastases-main/train/nnUNet_rain/nnUnet_dataset:/workspace/nnunet_data \
  nnunet-rain "nnUNetv2_train 1 3d_fullres 0 -num_gpus 2"
```

## 4. Inference (Prediction)
After training is complete, you can generate predictions on new, unseen data. You will need to map an additional input directory containing the new raw scans, and an output directory for the predictions.

```bash
docker run --gpus all -it --rm \
  -v /mnt/local/data/rainsun/metastases/Rain-BrainMetastases-main/train/nnUNet_rain/nnUnet_dataset:/workspace/nnunet_data \
  -v /path/to/server/input_scans:/workspace/input \
  -v /path/to/server/output_predictions:/workspace/output \
  nnunet-rain "nnUNetv2_predict -i /workspace/input -o /workspace/output -d 1 -c 3d_fullres -f 0"
```
*(Make sure to replace the `/path/to/server/...` paths with the actual paths to your test data)*