import matplotlib.pyplot as plt
import numpy as np
import torch as th
from road_vector_tokenizer import RoadVectorTokenizer
from agent_motion_tokenizer import AgentMotionTokenizer
from waymo_open_dataset.protos import scenario_pb2
import tensorflow as tf
import os
import glob

def visualize_scene(map_data, agent_data, save_path=None):
    """
    Visualizes Map Vectors + Agent Histories in the same frame.
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_facecolor('white')

    # 1. Plot Map Vectors (Grey background)
    vectors = map_data['road_vectors']
    mask = map_data['road_mask']
    
    if isinstance(vectors, th.Tensor): vectors = vectors.numpy()
    if isinstance(mask, th.Tensor): mask = mask.numpy()
    
    valid_vecs = vectors[mask]
    print(f"Plotting {len(valid_vecs)} map vectors...")
    
    for vec in valid_vecs:
        sx, sy, ex, ey = vec[:4]
        # Simple grey lines for map
        ax.plot([sx, ex], [sy, ey], color='gray', linewidth=0.5, alpha=0.5)

    # 2. Plot Agents
    agents = agent_data['agent_history'] # [N, T, D]
    agent_mask = agent_data['agent_mask']
    
    if isinstance(agents, th.Tensor): agents = agents.numpy()
    if isinstance(agent_mask, th.Tensor): agent_mask = agent_mask.numpy()
    
    valid_agents = agents[agent_mask]
    print(f"Plotting {len(valid_agents)} agents...")
    
    # Color cycle for agents
    colors = ['blue', 'green', 'orange', 'purple', 'cyan']
    
    for i, agent_seq in enumerate(valid_agents):
        # Get valid steps for this agent (valid flag is index 9)
        valid_steps = agent_seq[:, 9] > 0
        traj = agent_seq[valid_steps, :2] # [T_valid, 2] (x, y)
        
        if len(traj) == 0: continue
        
        # SDC is usually index 0 (we sorted it to top)
        is_sdc = (i == 0)
        color = 'red' if is_sdc else colors[i % len(colors)]
        zorder = 20 if is_sdc else 10
        
        # Plot Trajectory Trail
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=1.5, alpha=0.8, zorder=zorder)
        
        # Plot Current Position (Last step)
        curr_pos = traj[-1]
        marker = '>' if is_sdc else 'o' # SDC is arrow, others are dots
        size = 12 if is_sdc else 6
        ax.plot(curr_pos[0], curr_pos[1], marker=marker, color=color, markersize=size, zorder=zorder+1)

    # 3. Styling
    ax.set_title("Joint Tokenization: Map + Agents (Rotated Frame)")
    ax.set_xlabel("Local X (m)")
    ax.set_ylabel("Local Y (m)")
    ax.set_aspect('equal')
    
    # Zoom in
    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # --- Load Data ---
    DATASET_FOLDER = '/home/hansung/end2end_ad/datasets/waymo_open_dataset_motion_v_1_3_0/uncompressed/scenario/'
    VALIDATION_FILES = os.path.join(DATASET_FOLDER, 'validation/*.tfrecord*') 
    filenames = tf.io.matching_files(VALIDATION_FILES)
    dataset = tf.data.TFRecordDataset(filenames)
    raw_bytes = next(dataset.as_numpy_iterator())
    scenario = scenario_pb2.Scenario.FromString(raw_bytes)

    # --- Initialize Tokenizers ---
    map_tokenizer = RoadVectorTokenizer(max_dist=100.0, max_vectors=1024)
    agent_tokenizer = AgentMotionTokenizer(max_agents=32, history_steps=11)

    # --- Tokenize (WITH ROTATION) ---
    print("Tokenizing Map...")
    map_out = map_tokenizer.tokenize(scenario, current_step_index=10, rotate=True)
    
    print("Tokenizing Agents...")
    agent_out = agent_tokenizer.tokenize(scenario, current_step_index=10, rotate=True)

    # --- Visualize ---
    visualize_scene(map_out, agent_out, save_path="joint_viz.png")