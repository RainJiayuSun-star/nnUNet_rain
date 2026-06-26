#!/bin/bash
# run_test_pipeline.sh
# Sanity check/test script for nnU-Net v2 configurable training pipeline
# Run this INSIDE the running Docker container:
# e.g.: nnunet-rain "cd /opt/nnunet && chmod +x run_test_pipeline.sh && ./run_test_pipeline.sh"

# Stop script execution on error
set -e

# Configuration variables
DATASET_ID=1
CONFIGURATION="3d_fullres"
FOLD=0
TRAINER="nnUNetTrainerMetsConfigurable"
PLANS="nnUNetResEncUNetLPlans_custom_0624"
SUMMARY_FILE="training_plan_and_log/test_benchmark_summary.md"

# Results base directory (mounted path inside the Docker container)
RESULTS_BASE_DIR="/workspace/nnunet_data/nnUNet_results/Dataset001_UCSFbrainmets"

# Minimal test config file
CONFIGS=(
  "configs/exp_test.yaml"
)

echo "=== Starting PIPELINE TEST RUN ==="
echo "Logging test benchmark results to: $SUMMARY_FILE"

# Loop through each configuration file in the list
for CFG in "${CONFIGS[@]}"; do
  if [ ! -f "$CFG" ]; then
    echo "Error: Test configuration file '$CFG' does not exist!"
    exit 1
  fi

  echo "--------------------------------------------------"
  echo ">>> Launching Test Experiment: $CFG"
  echo "--------------------------------------------------"
  
  # 1. Copy the current experiment config to the default path read by the trainer
  cp "$CFG" train_config.yaml
  
  # 2. Run the nnU-Net training command
  nnUNetv2_train $DATASET_ID $CONFIGURATION $FOLD -p $PLANS -tr $TRAINER
  
  echo ">>> Test run complete. Parsing results..."
  
  # 3. Parse validation summary metrics and log to test summary file
  python parse_results.py train_config.yaml "$RESULTS_BASE_DIR" "$FOLD" "$SUMMARY_FILE"
  
done

echo "=== PIPELINE TEST RUN COMPLETE ==="
