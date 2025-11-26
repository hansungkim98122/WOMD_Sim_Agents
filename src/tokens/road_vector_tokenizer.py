from tokens.tokenizer import Tokenizer
import numpy as np
import torch as th
import pickle
from waymo_open_dataset.protos import map_pb2, scenario_pb2


# --- Expanded Mappings (same as before) ---
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
    def __init__(self,
                 max_dist: float = 200.0,
                 max_vectors: int = 2048,
                 max_segment_len: float = 5.0,
                 vocab_size: int = 1024,
                 use_rotation: bool = True,
                 use_y_flip_aug: bool = True):
        """
        Args:
            max_dist: max distance from center to keep map features.
            max_vectors: max number of vectors per scene (padding/truncation).
            max_segment_len: max length of each polyline segment (in meters).
            vocab_size: number of road-vector tokens in vocabulary.
            use_rotation: if False, never rotate map to SDC heading = 0,
                          regardless of the 'rotate' arg in methods.
            use_y_flip_aug: if False, never apply y-axis flip augmentation,
                            regardless of the 'augment' arg in tokenize().
        """
        super().__init__()
        self.max_dist = float(max_dist)
        self.max_vectors = int(max_vectors)
        self.max_segment_len = float(max_segment_len)

        self.vocab_size = int(vocab_size)
        self.vocab = None
        self.vocab_mean = None
        self.vocab_std = None
        self._vocab_norm = None

        # NEW: global switches
        self.use_rotation = bool(use_rotation)
        self.use_y_flip_aug = bool(use_y_flip_aug)


    # -------------------------------------------------------------------------
    # HIGH-LEVEL API
    # -------------------------------------------------------------------------

    def tokenize(self, scenario, current_step_index=10,
                 center_offset=None, rotate=True, augment=False,
                 return_token_ids=False, pad_token_id=0):
        """
        Converts WOMD Scenario into segmented road vectors.

        Args:
            scenario: scenario_pb2.Scenario or serialized bytes (same as build_vocabulary).
            current_step_index: SDC timestep used for centering / rotation.
            center_offset: optional [x,y] override center (world frame).
            rotate: if True, rotates map so SDC heading is 0.
            augment: if True, returns [2, N, 9] (original + flipped-y) (and masks).
            return_token_ids: if True AND vocab is available, also return
                              integer road token ids per vector.
            pad_token_id: id used for padded vectors when return_token_ids=True.

        Returns:
            Dict:
              'road_vectors': [B, N, 9] float32
              'road_mask':    [B, N]   bool
              optionally:
              'road_token_ids': [B, N] int64 (if return_token_ids=True and vocab exists)
        """
        scenario = self._ensure_scenario(scenario)

        # 1. base (original) scene
        base_vectors, base_mask = self._extract_single_scene(
            scenario, current_step_index, center_offset, rotate
        )  # [N, 9], [N]

        results_vec = [base_vectors]
        results_mask = [base_mask]
        results_tokens = []

        # 2. augmentation (flip Y)
        if augment:
            aug_vectors = base_vectors.clone()
            # flip sy, ey
            aug_vectors[:, 1] *= -1
            aug_vectors[:, 3] *= -1

            # flip turn direction semantics (2 <-> 3)
            turn_dirs = aug_vectors[:, 6]
            is_left = (turn_dirs == 2)
            is_right = (turn_dirs == 3)
            aug_vectors[is_left, 6] = 3
            aug_vectors[is_right, 6] = 2

            results_vec.append(aug_vectors)
            results_mask.append(base_mask.clone())

        # 3. optional discrete tokenization
        if return_token_ids and self.vocab is not None:
            for v, m in zip(results_vec, results_mask):
                ids = self.encode(v, mask=m, pad_token_id=pad_token_id)
                results_tokens.append(ids)

        out = {
            'road_vectors': th.stack(results_vec, dim=0),  # [B, N, 9]
            'road_mask': th.stack(results_mask, dim=0)     # [B, N]
        }
        if return_token_ids and self.vocab is not None:
            out['road_token_ids'] = th.stack(results_tokens, dim=0)  # [B, N]

        return out

    # -------------------------------------------------------------------------
    # VOCABULARY BUILDING (OFFLINE)
    # -------------------------------------------------------------------------

    def build_vocabulary_from_scenarios(self,
                                        scenarios,
                                        current_step_index: int = 10,
                                        center_offset=None,
                                        rotate: bool = True,
                                        max_scenarios: int = 1000,
                                        max_vectors_total: int = 300_000,
                                        kmeans_iters: int = 20,
                                        random_state: int = 0):
        """
        Offline: build road vector vocabulary from many scenarios.

        Args:
            rotate: per-call request to rotate to SDC heading 0.
                    Actual behavior is (rotate AND self.use_rotation).
        """
        rng = np.random.default_rng(random_state)
        collected = []
        n_vec = 0
        n_scen = 0

        # Effective rotation flag for vocab building
        do_rotate = bool(rotate) and self.use_rotation

        for item in scenarios:
            scenario = self._ensure_scenario(item)
            n_scen += 1

            vecs, mask = self._extract_single_scene(
                scenario, current_step_index, center_offset, do_rotate
            )
            vecs_np = vecs[mask].cpu().numpy()
            if vecs_np.size == 0:
                if max_scenarios is not None and n_scen >= max_scenarios:
                    break
                continue

            remaining = max_vectors_total - n_vec
            if remaining <= 0:
                break
            if vecs_np.shape[0] > remaining:
                idx = rng.choice(vecs_np.shape[0], size=remaining, replace=False)
                vecs_np = vecs_np[idx]

            collected.append(vecs_np)
            n_vec += vecs_np.shape[0]

            if max_scenarios is not None and n_scen >= max_scenarios:
                break
            if n_vec >= max_vectors_total:
                break

        if not collected:
            raise RuntimeError("No road vectors collected to build vocabulary.")

        X = np.vstack(collected).astype(np.float32)

        # Normalize features (z-score)
        self.vocab_mean = X.mean(axis=0, keepdims=True)
        self.vocab_std = X.std(axis=0, keepdims=True) + 1e-6
        Xn = (X - self.vocab_mean) / self.vocab_std  # [M, 9]

        # K-means clustering to get centers (k-disks-ish codebook)
        centers = self._kmeans(Xn, self.vocab_size,
                               iters=kmeans_iters, rng=rng)

        # store centers in both spaces
        self._vocab_norm = centers            # normalized centers
        self.vocab = centers * self.vocab_std + self.vocab_mean  # back to original

        print(f"[RoadVectorTokenizer] Built vocab with {self.vocab.shape[0]} tokens "
              f"from {n_vec} vectors ({n_scen} scenarios).")

    def save_smart_tokens(self, path: str) -> None:
        """
        Save vocab in SMART TokenProcessor map format (traj_src key).
        """
        if self.vocab is None:
            raise RuntimeError("Vocabulary not built or loaded yet.")
        payload = {"traj_src": np.asarray(self.vocab, dtype=np.float32)}
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def load_smart_tokens(self, path: str) -> None:
        """
        Load SMART-format map tokens and rebuild normalization stats.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        if "traj_src" not in data:
            raise ValueError("Expected key 'traj_src' in SMART map token pickle.")
        self.vocab = np.asarray(data["traj_src"], dtype=np.float32)
        self.vocab_mean = self.vocab.mean(axis=0, keepdims=True)
        self.vocab_std = self.vocab.std(axis=0, keepdims=True) + 1e-6
        self._vocab_norm = (self.vocab - self.vocab_mean) / self.vocab_std

    # -------------------------------------------------------------------------
    # ENCODING (CONTINUOUS → DISCRETE)
    # -------------------------------------------------------------------------

    def encode(self,
               vectors: th.Tensor,
               mask: th.Tensor = None,
               pad_token_id: int = 0) -> th.Tensor:
        """
        Map continuous road vectors to nearest vocabulary token IDs.

        Args:
            vectors: [N, 9] torch.Tensor or np.ndarray, same feature layout as tokenize().
            mask:   optional [N] bool mask; if provided, invalid positions get pad_token_id.
            pad_token_id: int to use where mask==False.

        Returns:
            torch.LongTensor [N] of token IDs in [0, vocab_size-1] (or pad_token_id where masked).
            (You can +1 if you want 0 as global pad id.)
        """
        if self.vocab is None or self._vocab_norm is None:
            raise RuntimeError("Vocabulary not built or loaded yet.")

        # Convert to numpy
        if isinstance(vectors, th.Tensor):
            X = vectors.detach().cpu().numpy()
        else:
            X = np.asarray(vectors, dtype=np.float32)

        if X.ndim != 2 or X.shape[1] != 9:
            raise ValueError(f"Expected vectors [N,9], got {X.shape}")

        # normalize with stored stats
        Xn = (X - self.vocab_mean) / self.vocab_std  # [N, 9]

        # compute squared L2 distance to each center
        # shapes: Xn [N,9], centers [K,9]
        diff = Xn[:, None, :] - self._vocab_norm[None, :, :]  # [N,K,9]
        d2 = (diff ** 2).sum(axis=-1)                         # [N,K]
        nearest = d2.argmin(axis=1).astype(np.int64)          # [N]

        if mask is not None:
            if isinstance(mask, th.Tensor):
                mask_np = mask.detach().cpu().numpy().astype(bool)
            else:
                mask_np = np.asarray(mask, dtype=bool)
            if mask_np.shape[0] != nearest.shape[0]:
                raise ValueError("mask and vectors must have same length.")
            out = np.full_like(nearest, pad_token_id, dtype=np.int64)
            out[mask_np] = nearest[mask_np]
        else:
            out = nearest

        return th.from_numpy(out).long()

    # -------------------------------------------------------------------------
    # INTERNAL HELPERS
    # -------------------------------------------------------------------------

    def _ensure_scenario(self, item):
        """
        Accepts:
          • scenario_pb2.Scenario
          • bytes / bytearray
          • dict with 'scenario/bytes' or 'scenario'
        Returns:
          scenario_pb2.Scenario
        """
        if isinstance(item, scenario_pb2.Scenario):
            return item

        if isinstance(item, (bytes, bytearray)):
            s = scenario_pb2.Scenario()
            s.ParseFromString(item)
            return s

        if isinstance(item, dict):
            if "scenario/bytes" in item:
                scen_bytes = item["scenario/bytes"]
            elif "scenario" in item:
                scen_bytes = item["scenario"]
            else:
                raise TypeError(f"Dict missing 'scenario/bytes' key: {item.keys()}")
            if isinstance(scen_bytes, np.ndarray):
                scen_bytes = scen_bytes.item()
            s = scenario_pb2.Scenario()
            s.ParseFromString(scen_bytes)
            return s

        raise TypeError(f"Unsupported scenario item type: {type(item)}")

    def _extract_single_scene(self, scenario, current_step_index, center_offset, rotate):
        """Parse + vectorize one scene once (same as Gemini, but reused)."""

        # 1. center & heading
        heading = 0.0
        if center_offset is None:
            sdc_track = scenario.tracks[scenario.sdc_track_index]
            state = sdc_track.states[current_step_index]
            center_point = np.array([state.center_x, state.center_y])
            heading = state.heading
        else:
            center_point = np.array(center_offset, dtype=np.float32)

        # 2. rotation matrix
        if rotate:
            c = np.cos(-heading)
            s = np.sin(-heading)
            rotation_matrix = np.array([[c, -s],
                                        [s,  c]])
        else:
            rotation_matrix = np.eye(2)

        # 3. dynamic states (traffic light)
        tl_states = self._extract_dynamic_states(scenario, current_step_index)

        all_vectors = []

        for feature in scenario.map_features:
            raw_points, type_id, extra_feats = self._parse_feature(feature)
            if raw_points is None or len(raw_points) < 2:
                continue

            pts_norm = raw_points - center_point  # center
            pts_norm = pts_norm @ rotation_matrix.T  # rotate

            # distance filter: keep features that come close to center
            if np.min(np.linalg.norm(pts_norm, axis=1)) > self.max_dist:
                continue

            # traffic light
            tl_state_int = 0
            if feature.id in tl_states:
                tl_state_int = TL_STATE_MAPPING.get(tl_states[feature.id], 0)

            # subdivide into short segments
            vectors = self._subdivide_and_vectorize(pts_norm)  # [S, 4] (sx,sy,ex,ey)
            num_segs = len(vectors)
            if num_segs == 0:
                continue

            f_type = np.full((num_segs, 1), type_id, dtype=np.float32)
            f_tl = np.full((num_segs, 1), tl_state_int, dtype=np.float32)
            f_turn = np.full((num_segs, 1), extra_feats.get('turn_dir', 0), dtype=np.float32)
            f_inter = np.full((num_segs, 1), int(extra_feats.get('is_intersection', False)),
                              dtype=np.float32)

            diffs = vectors[:, 2:4] - vectors[:, 0:2]
            lengths = np.linalg.norm(diffs, axis=1, keepdims=True).astype(np.float32)

            final_vecs = np.hstack([vectors.astype(np.float32),
                                    f_type, f_tl, f_turn, f_inter, lengths])
            all_vectors.append(final_vecs)

        # 4. pad/clip to max_vectors
        if not all_vectors:
            vecs = th.zeros((self.max_vectors, 9), dtype=th.float32)
            mask = th.zeros((self.max_vectors,), dtype=th.bool)
            return vecs, mask

        all_vectors = np.vstack(all_vectors)  # [M,9]

        # clip: keep closest to center if too many
        if len(all_vectors) > self.max_vectors:
            dists = np.linalg.norm(all_vectors[:, :2], axis=1)
            indices = np.argsort(dists)[:self.max_vectors]
            all_vectors = all_vectors[indices]

        num_valid = len(all_vectors)
        valid_mask = np.ones(num_valid, dtype=bool)

        if num_valid < self.max_vectors:
            pad_len = self.max_vectors - num_valid
            pad_vecs = np.zeros((pad_len, all_vectors.shape[1]), dtype=np.float32)
            pad_mask = np.zeros(pad_len, dtype=bool)
            all_vectors = np.vstack([all_vectors, pad_vecs])
            valid_mask = np.hstack([valid_mask, pad_mask])

        return th.from_numpy(all_vectors).float(), th.from_numpy(valid_mask).bool()

    def _subdivide_and_vectorize(self, points):
        """Break polylines into <= max_segment_len segments."""
        vectors = []
        for i in range(len(points) - 1):
            start = points[i]
            end = points[i + 1]
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
            if l.type == map_pb2.LaneCenter.TYPE_FREEWAY:
                type_id = MAP_TYPE_MAPPING['LANE_FREEWAY']
            elif l.type == map_pb2.LaneCenter.TYPE_BIKE_LANE:
                type_id = MAP_TYPE_MAPPING['LANE_BIKE_LANE']
            else:
                type_id = MAP_TYPE_MAPPING['LANE_SURFACE_STREET']
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
            points = np.array([[p.x, p.y], [p.x + 0.1, p.y]])  # tiny segment
            type_id = MAP_TYPE_MAPPING['STOP_SIGN']

        elif feature.HasField('crosswalk'):
            points = np.array([[p.x, p.y] for p in feature.crosswalk.polygon])
            if len(points) > 0:
                points = np.vstack([points, points[0]])
            type_id = MAP_TYPE_MAPPING['CROSSWALK']

        elif feature.HasField('speed_bump'):
            points = np.array([[p.x, p.y] for p in feature.speed_bump.polygon])
            if len(points) > 0:
                points = np.vstack([points, points[0]])
            type_id = MAP_TYPE_MAPPING['SPEED_BUMP']

        return points, type_id, extra

    def _infer_turn_direction(self, points):
        if len(points) < 2:
            return 0
        sample_dist = min(5, len(points) - 1)
        v_start = points[sample_dist] - points[0]
        v_end = points[-1] - points[-1 - sample_dist]
        ang_start = np.arctan2(v_start[1], v_start[0])
        ang_end = np.arctan2(v_end[1], v_end[0])
        diff = ang_end - ang_start
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        THRESHOLD = 0.52
        if diff > THRESHOLD:
            return 2  # left
        elif diff < -THRESHOLD:
            return 3  # right
        else:
            return 1  # straight

    def _extract_dynamic_states(self, scenario, step_index):
        tl_map = {}
        if step_index < len(scenario.dynamic_map_states):
            dms = scenario.dynamic_map_states[step_index]
            for ls in dms.lane_states:
                tl_map[ls.lane] = map_pb2.TrafficSignalLaneState.State.Name(ls.state)
        return tl_map

    # --- tiny k-means implementation (for vocab building) ---

    def _kmeans(self, X, k, iters=20, rng=None):
        """
        Simple k-means with k-means++ init.

        Args:
            X: [M,D] normalized data
            k: number of clusters
        """
        if rng is None:
            rng = np.random.default_rng()

        M, D = X.shape
        if k > M:
            k = M

        # k-means++ init
        centers = np.empty((k, D), dtype=np.float32)
        # 1st center: random
        idx0 = rng.integers(0, M)
        centers[0] = X[idx0]

        # remaining centers
        closest_d2 = np.full(M, np.inf, dtype=np.float32)
        for c in range(1, k):
            diff = X - centers[c - 1]
            d2 = (diff ** 2).sum(axis=1)
            closest_d2 = np.minimum(closest_d2, d2)
            probs = closest_d2 / closest_d2.sum()
            idx = rng.choice(M, p=probs)
            centers[c] = X[idx]

        # Lloyd iterations
        for _ in range(iters):
            # assign
            diff = X[:, None, :] - centers[None, :, :]
            d2 = (diff ** 2).sum(axis=2)
            labels = d2.argmin(axis=1)

            # update
            for c in range(k):
                mask = (labels == c)
                if np.any(mask):
                    centers[c] = X[mask].mean(axis=0)

        return centers.astype(np.float32)

if __name__ == "__main__":
    # Example usage: build road vector vocabulary from Waymo Open Dataset
    import os
    import tensorflow as tf
    from waymo_open_dataset.protos import scenario_pb2

    DATASET_FOLDER = '/home/hansung/end2end_ad/datasets/waymo_open_dataset_motion_v_1_3_0/uncompressed/scenario/'
    TRAIN_FILES = os.path.join(DATASET_FOLDER, 'training/training.tfrecord*')

    filenames = tf.io.matching_files(TRAIN_FILES)
    dataset = tf.data.TFRecordDataset(filenames)
    dataset_iterator = dataset.as_numpy_iterator()

    tokenizer = RoadVectorTokenizer(
        max_dist=1000.0,
        max_vectors=2048,
        max_segment_len=5.0,
        vocab_size=1024,
        use_rotation=False,      # never rotate to SDC heading 0
        use_y_flip_aug=False,    # never do y-flip augmentation
    )

    # Build vocabulary on (say) first 5k scenes and about 300k vectors.
    tokenizer.build_vocabulary_from_scenarios(
        dataset_iterator,
        current_step_index=10,
        rotate=True,
        max_scenarios=5000,
        max_vectors_total=300_000,
        kmeans_iters=20,
        random_state=0,
    )

    #pickle dump
    tokenizer.save_smart_tokens("map_traj_token5.pkl")
