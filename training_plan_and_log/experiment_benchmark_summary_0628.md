# Automated Experimentation Benchmark Summary

| Experiment Name | LR | WD | Epochs | Dropout | Region Weights | Mean Dice | Class 1 (Edema) | Class 2 (Necrosis) | Class 3 (Enhancing) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| exp_lr_2e-4_wd_3e-4 | 0.0002 | 0.0003 | 300 | 0.3 | [1.0, 1.0, 2.0] | 39.12% | N/A | N/A | N/A |
| exp_lr_1e-4_wd_5e-4 | 0.0001 | 0.0005 | 300 | 0.3 | [1.0, 1.0, 2.0] | 39.17% | N/A | N/A | N/A |
| exp_lr_2e-4_dropout_0.4 | 0.0002 | 0.0003 | 300 | 0.4 | [1.0, 1.0, 2.0] | 38.31% | N/A | N/A | N/A |
| exp_lr_3e-4_wd_1e-4_dropout_0.2 | 0.0003 | 0.0001 | 300 | 0.2 | [1.0, 1.0, 1.0] | 35.87% | N/A | N/A | N/A |
