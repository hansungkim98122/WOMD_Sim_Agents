from tokenizer import Tokenizer
import numpy as np
import torch as th
from waymo_open_dataset.protos import map_pb2

# --- Expanded Mappings ---
MAP_TYPE_MAPPING = {
    'LANE_FREEWAY': 1,
    'LANE_SURFACE_STREET': 2,
    'LANE_BIKE_LANE': 3,
    'ROAD_LINE_BROKEN_SINGLE_WHITE': 4,
    'ROAD_LINE_SOLID_SINGLE_WHITE': 5,
    'ROAD_LINE_SOLID_DOUBLE_WHITE': 6,
    'ROAD_LINE_BROKEN_SINGLE_YELLOW': 7,
    'ROAD_LINE_BROKEN_DOUBLE_YELLOW': 8,
    'ROAD_LINE_SOLID_SINGLE_YELLOW': 9,
    'ROAD_LINE_SOLID_DOUBLE_YELLOW': 10,
    'ROAD_LINE_PASSING_DOUBLE_YELLOW': 11,
    'ROAD_EDGE_BOUNDARY': 12,
    'ROAD_EDGE_MEDIAN': 13,
    'STOP_SIGN': 14,
    'CROSSWALK': 15,
    'SPEED_BUMP': 16,
}

TL_STATE_MAPPING = {
    'LANE_STATE_UNKNOWN': 0,
    'LANE_STATE_ARROW_STOP': 1,
    'LANE_STATE_ARROW_CAUTION': 2,
    'LANE_STATE_ARROW_GO': 3,
    'LANE_STATE_STOP': 4,
    'LANE_STATE_CAUTION': 5,
    'LANE_STATE_GO': 6,
    'LANE_STATE_FLASHING_STOP': 7,
    'LANE_STATE_FLASHING_CAUTION': 8,
}

class RoadVectorTokenizer(Tokenizer):
    def __init__(self, max_dist=200.0, max_vectors=2048, max_segment_len=5.0):
        super().__init__()
        self.max_dist = max_dist
        self.max_vectors = max_vectors
        self.max_segment_len = max_segment_len

    def tokenize(self, scenario, current_step_index=10, center_offset=None, rotate=True):
        """
        Converts WOMD Scenario into segmented road vectors.
        Returns vectors of shape [N, 9].
        
        Args:
            rotate (bool): If True, rotates map so SDC heading is 0 (facing +x).
                           If False, keeps global map orientation (only centers).
        """
        
        # 1. Determine Center & Heading
        heading = 0.0
        if center_offset is None:
            sdc_track = scenario.tracks[scenario.sdc_track_index]
            state = sdc_track.states[current_step_index]
            center_point = np.array([state.center_x, state.center_y])
            heading = state.heading
        else:
            center_point = np.array(center_offset)

        # 2. Create Rotation Matrix (Global -> Ego Frame)
        if rotate:
            # We rotate by -heading to align the world with the ego's forward direction.
            c = np.cos(-heading)
            s = np.sin(-heading)
            rotation_matrix = np.array([
                [c, -s], 
                [s,  c]
            ])
        else:
            # Identity matrix (No rotation)
            rotation_matrix = np.eye(2)

        # 3. Extract Dynamic States
        tl_states = self._extract_dynamic_states(scenario, current_step_index)

        all_vectors = []
        
        for feature in scenario.map_features:
            # Parse geometry and basic type
            raw_points, type_id, extra_feats = self._parse_feature(feature)
            
            if raw_points is None or len(raw_points) < 2:
                continue

            # A. Center
            pts_norm = raw_points - center_point
            
            # B. Rotate (or just Identity transform)
            pts_norm = pts_norm @ rotation_matrix.T
            
            # Distance Filter 
            if np.min(np.linalg.norm(pts_norm, axis=1)) > self.max_dist:
                continue

            # Extract Traffic Light State
            tl_state_int = 0
            if feature.id in tl_states:
                tl_state_int = TL_STATE_MAPPING.get(tl_states[feature.id], 0)

            # --- 4. SUBDIVIDE AND VECTORIZE ---
            vectors = self._subdivide_and_vectorize(pts_norm)
            
            # Add Semantic Features
            num_segs = len(vectors)
            f_type = np.full((num_segs, 1), type_id)
            f_tl = np.full((num_segs, 1), tl_state_int)
            f_turn = np.full((num_segs, 1), extra_feats.get('turn_dir', 0))
            f_inter = np.full((num_segs, 1), int(extra_feats.get('is_intersection', False)))
            
            diffs = vectors[:, 2:4] - vectors[:, 0:2]
            lengths = np.linalg.norm(diffs, axis=1, keepdims=True)

            final_vecs = np.hstack([vectors, f_type, f_tl, f_turn, f_inter, lengths])
            all_vectors.append(final_vecs)

        # 5. Stack and Pad
        if not all_vectors:
            return self._empty_result(feature_dim=9)

        all_vectors = np.vstack(all_vectors)
        
        # Clipping
        if len(all_vectors) > self.max_vectors:
            dists = np.linalg.norm(all_vectors[:, :2], axis=1)
            indices = np.argsort(dists)[:self.max_vectors]
            all_vectors = all_vectors[indices]

        # Padding
        num_valid = len(all_vectors)
        valid_mask = np.ones(num_valid, dtype=bool)
        
        if num_valid < self.max_vectors:
            pad_len = self.max_vectors - num_valid
            pad_vecs = np.zeros((pad_len, all_vectors.shape[1]))
            pad_mask = np.zeros(pad_len, dtype=bool)
            all_vectors = np.vstack([all_vectors, pad_vecs])
            valid_mask = np.hstack([valid_mask, pad_mask])

        return {
            'road_vectors': th.from_numpy(all_vectors).float(),
            'road_mask': th.from_numpy(valid_mask).bool()
        }

    def _subdivide_and_vectorize(self, points):
        vectors = []
        for i in range(len(points) - 1):
            start = points[i]
            end = points[i+1]
            vec = end - start
            dist = np.linalg.norm(vec)
            
            if dist <= self.max_segment_len:
                vectors.append(np.concatenate([start, end]))
            else:
                num_splits = int(np.ceil(dist / self.max_segment_len))
                step_vec = vec / num_splits
                curr = start
                for _ in range(num_splits):
                    next_p = curr + step_vec
                    vectors.append(np.concatenate([curr, next_p]))
                    curr = next_p
        return np.array(vectors)

    def _parse_feature(self, feature):
        points = []
        type_id = 0
        extra = {'turn_dir': 0, 'is_intersection': False}

        if feature.HasField('lane'):
            l = feature.lane
            points = np.array([[p.x, p.y] for p in l.polyline])
            if l.type == map_pb2.LaneCenter.TYPE_FREEWAY: type_id = MAP_TYPE_MAPPING['LANE_FREEWAY']
            elif l.type == map_pb2.LaneCenter.TYPE_BIKE_LANE: type_id = MAP_TYPE_MAPPING['LANE_BIKE_LANE']
            else: type_id = MAP_TYPE_MAPPING['LANE_SURFACE_STREET']
            extra['turn_dir'] = self._infer_turn_direction(points) 
            extra['is_intersection'] = l.interpolating 

        elif feature.HasField('road_line'):
            points = np.array([[p.x, p.y] for p in feature.road_line.polyline])
            type_id = MAP_TYPE_MAPPING.get('ROAD_LINE_BROKEN_SINGLE_WHITE', 4)

        elif feature.HasField('road_edge'):
            points = np.array([[p.x, p.y] for p in feature.road_edge.polyline])
            type_id = MAP_TYPE_MAPPING['ROAD_EDGE_BOUNDARY']

        elif feature.HasField('stop_sign'):
            p = feature.stop_sign.position
            points = np.array([[p.x, p.y], [p.x+0.1, p.y]]) 
            type_id = MAP_TYPE_MAPPING['STOP_SIGN']

        elif feature.HasField('crosswalk'):
            points = np.array([[p.x, p.y] for p in feature.crosswalk.polygon])
            if len(points) > 0: points = np.vstack([points, points[0]])
            type_id = MAP_TYPE_MAPPING['CROSSWALK']

        elif feature.HasField('speed_bump'):
            points = np.array([[p.x, p.y] for p in feature.speed_bump.polygon])
            if len(points) > 0: points = np.vstack([points, points[0]])
            type_id = MAP_TYPE_MAPPING['SPEED_BUMP']

        return points, type_id, extra

    def _infer_turn_direction(self, points):
        if len(points) < 2: return 0
        sample_dist = min(5, len(points) - 1)
        v_start = points[sample_dist] - points[0]
        v_end = points[-1] - points[-1 - sample_dist]
        ang_start = np.arctan2(v_start[1], v_start[0])
        ang_end = np.arctan2(v_end[1], v_end[0])
        diff = ang_end - ang_start
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        THRESHOLD = 0.52 
        if diff > THRESHOLD: return 2 
        elif diff < -THRESHOLD: return 3 
        else: return 1 

    def _extract_dynamic_states(self, scenario, step_index):
        tl_map = {}
        if step_index < len(scenario.dynamic_map_states):
            dms = scenario.dynamic_map_states[step_index]
            for ls in dms.lane_states:
                tl_map[ls.lane] = map_pb2.TrafficSignalLaneState.State.Name(ls.state)
        return tl_map

    def _empty_result(self, feature_dim=9):
        return {
            'road_vectors': th.zeros((self.max_vectors, feature_dim)),
            'road_mask': th.zeros((self.max_vectors)).bool()
        }

if __name__ == "__main__":
    # Load a scenario (mocking the loading part)
    # scenario = ... load from tfrecord ...

    tokenizer = RoadVectorTokenizer(max_dist=int(300), max_vectors=int(1024))
    import os
    from waymo_open_dataset.protos import scenario_pb2
    import tensorflow as tf

    DATASET_FOLDER = '/home/hansung/end2end_ad/datasets/waymo_open_dataset_motion_v_1_3_0/uncompressed/scenario/'
    VALIDATION_FILES = os.path.join(DATASET_FOLDER, 'validation/validation.tfrecord*')
    filenames = tf.io.matching_files(VALIDATION_FILES)
    dataset = tf.data.TFRecordDataset(filenames)
    dataset_iterator = dataset.as_numpy_iterator()
    bytes_example = next(dataset_iterator)

    scenario = scenario_pb2.Scenario.FromString(bytes_example)

    # Tokenize
    output = tokenizer.tokenize(scenario, current_step_index=10,rotate=True)

    # Result is ready for your PyTorch model
    print("Vectors Shape:", output['road_vectors'].shape)