#!/bin/bash
# run_experiments_pipeline.sh
# Sequential automated training pipeline for nnU-Net v2 experiments
# This script should be executed INSIDE the running Docker container:
# e.g.: nnunet-rain "cd /opt/nnunet && ./run_experiments_pipeline.sh"

# Stop script execution on error
set -e

# Configuration variables
DATASET_ID=1
CONFIGURATION="3d_fullres"
FOLD=0
TRAINER="nnUNetTrainerMetsConfigurable"
PLANS="nnUNetResEncUNetLPlans_custom_0624"
SUMMARY_FILE="training_plan_and_log/experiment_benchmark_summary.md"

# Results base directory (mounted path inside the Docker container)
RESULTS_BASE_DIR="/workspace/nnunet_data/nnUNet_results/Dataset001_UCSFbrainmets"

# Pre-defined list of configuration YAML files you want to train sequentially
# Create a 'configs/' folder and populate it with your parameter files on the host
CONFIGS=(
  "configs/exp_lr_2e-4_wd_3e-4.yaml"
  "configs/exp_lr_1e-4_wd_5e-4.yaml"
  "configs/exp_lr_2e-4_dropout_0.4.yaml"
)

echo "=== Starting Automated nnU-Net Experimentation Pipeline ==="
echo "Logging benchmark results to: $SUMMARY_FILE"

# Loop through each configuration file in the list
for CFG in "${CONFIGS[@]}"; do
  if [ ! -f "$CFG" ]; then
    echo "Warning: Configuration file '$CFG' does not exist. Skipping."
    continue
  fi

  echo "--------------------------------------------------"
  echo ">>> Launching Experiment: $CFG"
  echo "--------------------------------------------------"
  
  # 1. Copy the current experiment config to the default path read by the trainer
  cp "$CFG" train_config.yaml
  
  # 2. Run the nnU-Net training command
  # (nnU-Net automatically runs validation on the fold cases at the end of training)
  nnUNetv2_train $DATASET_ID $CONFIGURATION $FOLD -p $PLANS -tr $TRAINER
  
  echo ">>> Experiment complete. Parsing results..."
  
  # 3. Parse validation summary metrics and log to summary file
  python parse_results.py train_config.yaml "$RESULTS_BASE_DIR" "$FOLD" "$SUMMARY_FILE"
  
done

echo "=== Automated Pipeline Complete! ==="
