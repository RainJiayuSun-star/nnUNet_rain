# nnU-Net Training Plan & Configuration

This document outlines the training strategy and hyperparameter configurations for the Brain Metastases segmentation models, specifically tailored to maximize the use of our high-performance hardware (e.g., 4x NVIDIA L40 GPUs with 40GB VRAM each).

## 1. Core Philosophy: Why No Hyperparameter Tuning?
nnU-Net is designed to eliminate the need for manual hyperparameter tuning (learning rate, optimizer, batch size, etc.). Its preprocessing pipeline dynamically calculates the optimal network architecture (patch size, batch size, pooling operations) based on the image properties of the dataset.

The fixed hyperparameter defaults (hardcoded in `nnUNetTrainer.py`) are:
- **Epochs:** 1000
- **Optimizer:** SGD with Nesterov momentum (0.99)
- **Initial Learning Rate:** 0.01 (decayed using a polynomial schedule)
- **Loss Function:** Dice + CrossEntropy

**Rule of Thumb:** Do not change the learning rate or epoch count unless running highly specialized, short ablation studies.

## 2. Standard Training Execution
Once the data is preprocessed (`nnUNetv2_plan_and_preprocess`), a standard training job on a single GPU is executed via:
```bash
docker run --gpus all -it --rm \
  -v /mnt/local/data/rainsun/metastases/Rain-BrainMetastases-main/train/nnUNet_rain/nnUnet_dataset:/workspace/nnunet_data \
  nnunet-rain "nnUNetv2_train 1 3d_fullres 0"
```
*(Where `1` is Dataset ID, `3d_fullres` is the model config, and `0` is the fold)*

---

## 3. High-Performance Hardware Scaling (Recommended Plan)
To fully utilize the 4x L40 GPUs, the following advanced configurations should be used.

### A. Distributed Data Parallel (DDP) Multi-GPU Training
nnU-Net defaults to 1 GPU. To drastically speed up the 1000 epochs, force it to use all available GPUs by defining the CUDA devices and passing the `-num_gpus` flag.
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 nnUNetv2_train 1 3d_fullres 0 -num_gpus 4
```

### B. Residual Encoder Architectures
The standard nnU-Net utilizes a plain U-Net. However, with 40GB of VRAM per GPU, we can afford much heavier models. The new **Residual Encoder** trainers provide significantly higher accuracy at the cost of compute.
Use the `-tr` (trainer) flag to invoke the large residual architectures:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 nnUNetv2_train 1 3d_fullres 0 -tr nnUNetTrainerResencL -num_gpus 4
```

### C. Advanced VRAM Targeting (Planning Phase)
By default, `nnUNetv2_plan_and_preprocess` targets an 8GB GPU, leading to conservative patch sizes. If we want to force the network to "see" a massive context window of the brain in a single pass, we can tell the planner to target our 40GB GPUs by setting environment variables *before* running preprocessing:
```bash
export nnUNet_keep_files_open=True
export nnUNet_vram_target_GB=35  # Target 35GB of our 40GB L40s

nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
```
*Note: This will rewrite `nnUNetPlans.json` with massive patch dimensions. Only do this if standard planning yields patch sizes that are too small.*

## 4. Training Logs & Evaluation
During training, progress is automatically tracked inside `nnUNet_results/Dataset001_UCSFbrainmets/...`.
- `training_log_*.txt`: Real-time text output of loss and pseudo-Dice scores.
- `progress.png`: A live-updating visual graph of training loss vs validation accuracy.
- `validation/summary.json`: The final definitive metrics (Dice, HD95) calculated after the 1000 epochs complete.
