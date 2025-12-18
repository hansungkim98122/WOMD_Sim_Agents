# Waymo Open Dataset 2025 Sim Agents Challenge
This repository is adopted from [SMART](https://github.com/rainmaker22/SMART ) with my own modification for implementation of the SMART backbone training, post-training, and evaluation. It includes the full pipeline from data preprocessing to WOD Sim Agent challenge submission. 

## Simulation Rollout Examples:
https://github.com/user-attachments/assets/4f08af72-8b16-481d-a2c7-e192ceb821b6

https://github.com/user-attachments/assets/5cdbad6d-b011-453b-bdcc-12c285da201f

https://github.com/user-attachments/assets/49478526-15c1-4e2b-acd3-0a1631addfd8

https://github.com/user-attachments/assets/f0abb8e2-83b2-4185-aa5a-fabd18829ca8

## Environment Setup:
The environment.yaml file that I used on my workstation is also included. Try creating a conda environment
```
conda create -f environment.yaml
conda activate e2e
```

```
pip install -r requirements.txt
```

See [SMART](https://github.com/rainmaker22/SMART ) for detailed environment setup and dependencies setup.

For your reference, training on NVIDIA RTX PRO 6000 Blackwell with 98GB VRAM took ~6 hours per epoch on the WOMD training dataset.
Fine-tuning took ~40 hours.

## Data preprocessing:
After downloading the Waymo Open Motion Dataset (Scenario: training, validation, and testing). You can start preprocessing data
```
cd src
```
```
python data_preprocessing.py --input_dir <raw_scenario_data_directory> --output_dir <desired output directory>
```
## Pre-training:
```
python train_smart.py
```

## Post-training: Fine-tuning using Reinforcement Learning
```
python rl_finetune.py
```

## Evaluation: 
```
python val_smart.py
```

## Rollout and Visualization:
See
```
src/validation_visualization.ipynb
```
## Submission Generation:
See 
```
src/submission_generation.ipynb
```
