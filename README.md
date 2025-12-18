# Waymo Open Dataset 2025 Sim Agents Challenge
This repository is adopted from [SMART](https://github.com/rainmaker22/SMART ) with my modifications to support training, post-training, and evaluation of the SMART backbone. SMART formulates the traffic agent (Vehicle, Pedestrians, Cyclists) motion prediction problem as a next token prediction. With recent advances and great success in text token prediction tasks using LLMs, motion prediction can borrow ideas from LLMs in their architectures and training algorithms. First, motion trajectories can be broken down into smaller units (~0.5 second segments) and discretized into finite motion token vocabularies. Using these discrete motion tokens, we can encode and decode agents' motion in the scene. Similarly, map information (polylines, polygons) can be tokenized. Given the map and trajectory history of the agents in the scene, the attention-based encoder-decoder architecture of SMART autoregressively generates agents' future trajectories by "predicting next tokens".

Because our token predictor outputs logits over a discrete token vocabulary, we can sample from the resulting categorical distribution to inject controlled randomness into trajectory generation. This produces diverse joint rollouts of the scene and makes the model well-suited for multi-agent simulation.

This repository provides the full pipeline from data preprocessing to WOD Sim Agent challenge submission for a SMART-backbone architecture. 

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
## Data Downloading:
You can download data from Google Cloud Buckets from [Waymo](https://waymo.com/open/download) website. Easiest way to download the data is to install gsutil and use the command
```
gsutil cp -r <bucket address. starts with gs:// > <destination folder directory>
```

## Data preprocessing:
After downloading the Waymo Open Motion Dataset (Scenario: training, validation, and testing). You can start preprocessing data, which generates pickle files for each scenario in the raw data proto.
```
cd src
```
```
python data_preprocessing.py --input_dir <raw_scenario_data_directory> --output_dir <desired output directory>
```

You can store the data in any directory you prefer, but you must explicitly set the paths to the raw (uncompressed) WOMD data and the preprocessed data in the YAML files under src/configs and in the path definitions used in the Jupyter notebooks.

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

# Acknowledgement
The majority of the code is from [SMART](https://github.com/rainmaker22/SMART)

# License
All code in this repository is licensed under the Apache License 2.0.
