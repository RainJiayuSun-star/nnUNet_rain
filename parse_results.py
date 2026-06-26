# parse_results.py
import os
import json
import yaml
import sys

def parse_and_log(config_path, results_base_dir, fold, summary_file):
    # 1. Load the experiment configuration to extract parameters
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    experiment_name = cfg.get("experiment_name", "default")
    initial_lr = cfg.get("initial_lr", "default")
    weight_decay = cfg.get("weight_decay", "default")
    num_epochs = cfg.get("num_epochs", "default")
    dropout_p = cfg.get("dropout_p", "default")
    region_weights = cfg.get("region_weights", "default")
    
    # 2. Locate output validation folder dynamically using the experiment_name
    matching_dirs = [d for d in os.listdir(results_base_dir) if f"_{experiment_name}__" in d]
    if not matching_dirs:
        print(f"Error: No results directories found matching experiment_name '{experiment_name}' under {results_base_dir}")
        return
    
    # Sort matching directories to use the most recently modified one if there are multiples
    matching_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(results_base_dir, x)), reverse=True)
    results_folder = os.path.join(results_base_dir, matching_dirs[0], f"fold_{fold}")
    summary_path = os.path.join(results_folder, "validation", "summary.json")
    
    if not os.path.exists(summary_path):
        print(f"Error: Validation summary not found at {summary_path}. The training run may have crashed.")
        mean_dice, c1, c2, c3 = "Crash/None", "N/A", "N/A", "N/A"
    else:
        with open(summary_path, 'r') as f:
            summary = json.load(f)
            
        metrics = summary.get("foreground_mean", summary.get("mean", {}))
        mean_dice = metrics.get("Dice", "N/A")
        c1 = summary.get("metric_per_class", {}).get("1", {}).get("Dice", "N/A")
        c2 = summary.get("metric_per_class", {}).get("2", {}).get("Dice", "N/A")
        c3 = summary.get("metric_per_class", {}).get("3", {}).get("Dice", "N/A")

    # Format dice scores as percentages
    def fmt(val):
        try:
            return f"{float(val)*100:.2f}%"
        except (ValueError, TypeError):
            return str(val)

    # 3. Create parent directories for summary file if needed
    summary_dir = os.path.dirname(summary_file)
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)

    # 4. Append row to centralized benchmark markdown file
    file_exists = os.path.exists(summary_file)
    with open(summary_file, 'a+') as f:
        if not file_exists or os.path.getsize(summary_file) == 0:
            f.write("# Automated Experimentation Benchmark Summary\n\n")
            f.write("| Experiment Name | LR | WD | Epochs | Dropout | Region Weights | Mean Dice | Class 1 (Edema) | Class 2 (Necrosis) | Class 3 (Enhancing) |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        
        f.write(f"| {experiment_name} | {initial_lr} | {weight_decay} | {num_epochs} | {dropout_p} | {region_weights} | {fmt(mean_dice)} | {fmt(c1)} | {fmt(c2)} | {fmt(c3)} |\n")
        
    print(f"Successfully logged results for '{experiment_name}' to {summary_file}")

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python parse_results.py <config_path> <results_base_dir> <fold> <summary_file>")
        sys.exit(1)
    parse_and_log(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
