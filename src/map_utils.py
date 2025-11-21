import matplotlib.pyplot as plt
import numpy as np
from waymo_open_dataset.protos import map_pb2
import torch as th
import matplotlib.cm as cm

def visualize_raw_map(scenario, title="Raw Map Polylines", save_path=None, rotate=True):
    """
    Visualizes the raw map features.
    
    Args:
        rotate (bool): If True, rotates map so SDC heading is 0 (facing +x).
                       If False, keeps global orientation (centered on SDC).
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_facecolor('white')

    # 1. Get SDC State
    sdc_id = scenario.sdc_track_index
    center_x, center_y = 0, 0
    heading = 0.0
    
    if sdc_id >= 0 and sdc_id < len(scenario.tracks):
        state = scenario.tracks[sdc_id].states[10] # 1.0s mark
        center_x, center_y = state.center_x, state.center_y
        heading = state.heading

    # 2. Configure Rotation
    if rotate:
        # Rotate world so Ego heading is 0 (Positive X)
        c = np.cos(-heading)
        s = np.sin(-heading)
        rotation_matrix = np.array([[c, -s], [s, c]])
        
        # Ego Marker: Red Triangle pointing Right (+X)
        ax.plot(0, 0, color='red', marker='>', markersize=12, label='Ego Vehicle', zorder=10)
        plot_title = f"{title}\n(Rotated: Ego facing +X)"
    else:
        # No rotation, just centering
        rotation_matrix = np.eye(2)
        
        # Ego Marker: Arrow pointing in actual heading direction
        dx = np.cos(heading) * 5.0
        dy = np.sin(heading) * 5.0
        ax.arrow(0, 0, dx, dy, color='red', width=0.5, head_width=2.0, zorder=10, label='Ego Vehicle')
        ax.plot(0, 0, 'ro', markersize=5) # Center dot
        plot_title = f"{title}\n(Global Orientation)"

    # 3. Iterate through all map features
    print(f"Plotting {len(scenario.map_features)} raw map features...")
    
    plotted_labels = set()
    plotted_labels.add('Ego Vehicle')

    for feature in scenario.map_features:
        points = None
        style = {'color': 'blue', 'linewidth': 0.5, 'alpha': 0.6, 'markersize': 0.5, 'marker': '.'}
        label = None

        # Extract geometry
        if feature.HasField('lane'):
            points = np.array([[p.x, p.y] for p in feature.lane.polyline])
            style['color'] = 'navy'; style['linewidth'] = 0.8; label = 'Lane Center'
        elif feature.HasField('road_line'):
            points = np.array([[p.x, p.y] for p in feature.road_line.polyline])
            style['color'] = 'gray'; style['linestyle'] = '--'
            label = 'Road Line'
        elif feature.HasField('road_edge'):
            points = np.array([[p.x, p.y] for p in feature.road_edge.polyline])
            style['color'] = 'black'; style['linewidth'] = 1.5; label = 'Road Edge'
        elif feature.HasField('stop_sign'):
            p = feature.stop_sign.position
            points = np.array([[p.x, p.y]]) 
            current_label = 'Stop Sign'
            lbl_arg = current_label if current_label not in plotted_labels else None
            
            # Transform Single Point
            points = points - np.array([center_x, center_y])
            points = points @ rotation_matrix.T
            
            ax.plot(points[0,0], points[0,1], 'rx', markersize=8, label=lbl_arg)
            if lbl_arg: plotted_labels.add(current_label)
            continue

        elif feature.HasField('crosswalk'):
            points = np.array([[p.x, p.y] for p in feature.crosswalk.polygon])
            style['color'] = 'purple'; style['alpha'] = 0.3; label = 'Crosswalk'
            if len(points) > 0: points = np.vstack([points, points[0]])
        elif feature.HasField('speed_bump'):
            points = np.array([[p.x, p.y] for p in feature.speed_bump.polygon])
            style['color'] = 'orange'; style['alpha'] = 0.5; label = 'Speed Bump'
            if len(points) > 0: points = np.vstack([points, points[0]])

        # 4. Transform and Plot
        if points is not None and len(points) > 0:
            # Center
            points = points - np.array([center_x, center_y])
            # Rotate
            points = points @ rotation_matrix.T
            
            if label and label not in plotted_labels:
                ax.plot(points[:, 0], points[:, 1], label=label, **style)
                plotted_labels.add(label)
            else:
                ax.plot(points[:, 0], points[:, 1], **style)

    window_size = 80 
    ax.set_xlim(-window_size, window_size)
    ax.set_ylim(-window_size, window_size)
    ax.set_aspect('equal')
    ax.set_title(f"{plot_title}\nScenario ID: {scenario.scenario_id}")
    ax.grid(True, alpha=0.2)
    ax.legend(loc='upper right', framealpha=0.9, shadow=True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()

def visualize_vector_tokens(road_vectors, road_mask, save_path=None, ego_heading=None):
    """
    Visualizes the vectorized map.
    
    Args:
        ego_heading: If None, assumes vectors are already rotated (Ego facing +X).
                     If provided (in radians), assumes vectors are Global and draws Ego arrow.
    """
    if isinstance(road_vectors, th.Tensor):
        vectors = road_vectors.detach().cpu().numpy()
        mask = road_mask.detach().cpu().numpy()
    else:
        vectors = road_vectors
        mask = road_mask

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_facecolor('white')

    valid_vectors = vectors[mask]
    num_vectors = len(valid_vectors)
    print(f"Visualizing {num_vectors} discrete vector tokens...")

    colors = cm.turbo(np.linspace(0, 1, num_vectors))
    np.random.seed(42); np.random.shuffle(colors)

    for i, vec in enumerate(valid_vectors):
        sx, sy, ex, ey = vec[:4]
        ax.plot([sx, ex], [sy, ey], color=colors[i], linewidth=1.5, alpha=0.9, marker='.', markersize=3)

    # Plot Ego Vehicle
    if ego_heading is None:
        # Assumed Rotated: Ego faces +X
        ax.plot(0, 0, marker='>', color='red', markersize=12, label='Ego Vehicle')
        title_suffix = "(Rotated)"
    else:
        # Assumed Global: Draw Arrow
        dx = np.cos(ego_heading) * 5.0
        dy = np.sin(ego_heading) * 5.0
        ax.arrow(0, 0, dx, dy, color='red', width=0.5, head_width=2.0, label='Ego Vehicle')
        ax.plot(0, 0, 'ro', markersize=5)
        title_suffix = "(Global Orientation)"

    ax.set_aspect('equal')
    ax.grid(False)
    ax.set_title(f"Road Vector Tokenization {title_suffix}\n({num_vectors} segments)")
    ax.set_xlabel("Local X (meters)")
    ax.set_ylabel("Local Y (meters)")
    
    zoom_range = 80
    ax.set_xlim(-zoom_range, zoom_range)
    ax.set_ylim(-zoom_range, zoom_range)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()

# --- Example Usage Block ---
if __name__ == "__main__":
    import tensorflow as tf
    from waymo_open_dataset.protos import scenario_pb2
    import glob
    import os
    from road_vector_tokenizer import RoadVectorTokenizer

    DATASET_FOLDER = '/home/hansung/end2end_ad/datasets/waymo_open_dataset_motion_v_1_3_0/uncompressed/scenario/'
    VALIDATION_FILES = os.path.join(DATASET_FOLDER, 'validation/*.tfrecord*') 
    
    filenames = tf.io.matching_files(VALIDATION_FILES)
    
    if len(filenames) > 0:
        dataset = tf.data.TFRecordDataset(filenames)
        iterator = dataset.as_numpy_iterator()
        raw_bytes = next(iterator)
        scenario = scenario_pb2.Scenario.FromString(raw_bytes)
        
        # Get heading for the non-rotated visualization
        sdc_track = scenario.tracks[scenario.sdc_track_index]
        heading = sdc_track.states[10].heading

        # ---------------------------------------------------------
        # CASE 1: Rotated (Standard for Training)
        # ---------------------------------------------------------
        print("--- Visualizing Case 1: Rotated ---")
        visualize_raw_map(scenario, save_path="raw_map_rotated.png", rotate=True)
        
        tokenizer = RoadVectorTokenizer(max_dist=200.0, max_vectors=2048)
        output_rot = tokenizer.tokenize(scenario, current_step_index=10, rotate=True)
        
        visualize_vector_tokens(output_rot['road_vectors'], output_rot['road_mask'], 
                              save_path="vectors_rotated.png", ego_heading=None)

        # ---------------------------------------------------------
        # CASE 2: Not Rotated (Global Orientation - For Ablation)
        # ---------------------------------------------------------
        print("--- Visualizing Case 2: Global Orientation ---")
        visualize_raw_map(scenario, save_path="raw_map_global.png", rotate=False)
        
        output_global = tokenizer.tokenize(scenario, current_step_index=10, rotate=False)
        
        visualize_vector_tokens(output_global['road_vectors'], output_global['road_mask'], 
                              save_path="vectors_global.png", ego_heading=heading)

    else:
        print("No files found.")