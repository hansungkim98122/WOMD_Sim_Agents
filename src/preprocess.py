import os
import glob
import numpy as np
import yaml
import torch as th
import tensorflow as tf
from tqdm import tqdm
from dataset_utils import load_dataset
from waymo_open_dataset.protos import scenario_pb2
from road_vector_tokenizer import RoadVectorTokenizer

# Hide GPU from TF to prevent memory conflicts, we only need CPU for this
tf.config.set_visible_devices([], 'GPU')

def preprocess_dataset(dataset, output_dir, tokenizer):
    """
    Reads TFRecords, tokenizes map/agents, and saves individual .pt files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate through raw bytes
    # NOTE: This assumes 'dataset' yields raw serialized bytes. 
    # If load_dataset yields a dictionary of features, this loop needs adjustment.
    for i, raw_bytes in tqdm(enumerate(dataset.as_numpy_iterator()), desc=f"Processing..."):
        
        try:
            # 1. Parse Proto
            # If raw_bytes is a dictionary (common in Waymo loaders), you might need to extract the raw string
            # But assuming standard raw loading:
            scenario = scenario_pb2.Scenario.FromString(raw_bytes)
            scenario_id = scenario.scenario_id
            
            # 2. Tokenize Map
            map_output = tokenizer.tokenize(scenario, current_step_index=10)
            
            # 3. Prepare Data Dictionary
            # We strictly keep these as Torch Tensors (CPU)
            data_payload = {
                'road_vectors': map_output['road_vectors'],
                'road_mask': map_output['road_mask'],
                # Add future agent features here:
                # 'agent_past': ...
                # 'agent_future': ...
            }
            
            # 4. Save to disk as .pt
            save_path = os.path.join(output_dir, f"{scenario_id}.pt")
            th.save(data_payload, save_path)
            
        except Exception as e:
            print(f"Error processing index {i}: {e}")
            continue

if __name__ == "__main__":
    # Configuration
    with open("dataset_configs/preprocess.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # Ensure load_dataset returns the RAW dataset (not parsed features)
    # if possible, otherwise you might need to bypass load_dataset and use tf.data.TFRecordDataset directly
    training_dataset, validation_dataset, test_dataset, features_description = load_dataset(cfg['raw_data_path'])

    # Initialize Tokenizers
    tokenizer = RoadVectorTokenizer(max_dist=cfg['max_dist'], max_vectors=cfg['max_vectors']) 

    datasets = [training_dataset, validation_dataset, test_dataset]
    dataset_types = ['train', 'validation', 'test']

    for dataset, split_name in zip(datasets, dataset_types):
        # Skip if dataset is None (e.g., test set might not be available)
        if dataset is None:
            continue
            
        OUTPUT_PATH = os.path.join(cfg['output_path'], split_name)
        
        print(f"Starting preprocessing for {split_name} dataset...")
        print(f"Saving to: {OUTPUT_PATH}")
        
        preprocess_dataset(dataset, OUTPUT_PATH, tokenizer)

    print("All datasets processed!")