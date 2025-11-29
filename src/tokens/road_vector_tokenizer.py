from tokenizer import Tokenizer
import numpy as np
import torch as th
import pickle
from waymo_open_dataset.protos import map_pb2, scenario_pb2


class RoadVectorTokenizer(Tokenizer):
    """
    SMART-style road tokenizer.

    Goal:
      - From many Waymo scenarios, extract local road polylines
        in SDC frame.
      - Resample each polyline to a fixed length `traj_len`
        (e.g., 11 points in 2D).
      - Run k-means over these (traj_len, 2) polylines to
        build a set of prototype road tokens.
      - Save as:
            {"traj_src": traj_src}
        where traj_src has shape (num_tokens, traj_len, 2).
    """

    def __init__(
        self,
        max_dist: float = 200.0,
        max_vectors: int = 2048,
        traj_len: int = 11,
        vocab_size: int = 1024,
        use_rotation: bool = True,
        max_segment_len: float = 5.0, # Kept for potential pre-segmentation
    ):
        super().__init__()
        self.max_dist = float(max_dist)
        # We'll still call this max_vectors for compatibility,
        # but it now means "max polylines per scene".
        self.max_vectors = int(max_vectors)

        self.traj_len = int(traj_len)   # SMART uses 11
        self.vocab_size = int(vocab_size)
        self.max_segment_len = float(max_segment_len)

        self.use_rotation = bool(use_rotation)

        # Vocabulary in trajectory space:
        #   self.vocab_traj: (K, traj_len, 2)
        # For clustering we also keep the flattened version.
        self.vocab_traj = None      # (K, traj_len, 2)
        self._vocab_flat = None     # (K, traj_len*2)

    # ----------------------------------------------------------------------
    # HIGH-LEVEL API
    # ----------------------------------------------------------------------
    def encode(self, vectors, mask=None, pad_token_id=0):
        """
        Encode raw road vectors into discrete token IDs using the built vocabulary.
        
        Args:
            vectors: [B, N, traj_len, 2] 
            mask: [B, N]
        """
        if self.vocab_traj is None:
            raise RuntimeError("Vocab not loaded")
            
        # Flatten input: [B, N, traj_len*2]
        B, N, T, D = vectors.shape
        flat_input = vectors.view(B, N, -1) 
        
        # Flatten vocab: [K, traj_len*2]
        if self._vocab_flat is None:
             self._vocab_flat = self.vocab_traj.reshape(self.vocab_size, -1)
             
        # Compute distances
        # We can do this per batch to save memory
        # ... (Implementation omitted for brevity, similar to before)
        return

    def tokenize(
        self,
        scenario,
        current_step_index: int = 10,
        center_offset=None,
        rotate: bool = True,
    ):
        """
        Extract road trajectories from a single scenario.

        Returns:
            {
              "road_traj":  Tensor [N, traj_len, 2],
              "road_mask":  Tensor [N] (bool)
            }
        """
        scenario = self._ensure_scenario(scenario)
        do_rotate = bool(rotate) and self.use_rotation

        with th.no_grad():
            trajs, mask = self._extract_single_scene(
                scenario, current_step_index, center_offset, do_rotate
            )

        return {
            "road_traj": trajs,   # [N, traj_len, 2] float32
            "road_mask": mask,    # [N] bool
        }

    # ----------------------------------------------------------------------
    # VOCABULARY BUILDING (OFFLINE)
    # ----------------------------------------------------------------------
    def build_vocabulary_from_scenarios(
        self,
        scenarios,
        current_step_index: int = 10,
        center_offset=None,
        rotate: bool = True,
        max_scenarios: int = 5000,
        max_vectors_total: int = 200_000,
        kmeans_iters: int = 20,
        random_state: int = 42,
    ):
        """
        Build SMART-style trajectory vocabulary from a stream of scenarios.
        """
        import gc

        rng = np.random.default_rng(random_state)

        # Reservoir of flattened trajectories: [K, traj_len*2]
        dim = self.traj_len * 2
        reservoir = np.zeros((max_vectors_total, dim), dtype=np.float32)
        reservoir_count = 0
        vectors_seen_total = 0
        scenarios_seen = 0

        do_rotate = bool(rotate) and self.use_rotation

        print(
            f"[RoadVectorTokenizer] Starting trajectory reservoir sampling "
            f"(max_vectors_total={max_vectors_total}, traj_len={self.traj_len})"
        )

        # Iterate over file paths (manual iterator management)
        # scenarios arg is expected to be a LIST of file paths based on our previous fix
        if hasattr(scenarios, 'numpy'): 
             scenarios = [f.numpy() for f in scenarios]
             
        import tensorflow as tf
        
        for file_path in scenarios:
            if max_scenarios and scenarios_seen >= max_scenarios:
                break
                
            dataset = tf.data.TFRecordDataset(file_path)
            
            for raw_bytes in dataset.as_numpy_iterator():
                try:
                    scenario = scenario_pb2.Scenario()
                    scenario.ParseFromString(raw_bytes)
                    
                    with th.no_grad():
                        trajs, mask = self._extract_single_scene(
                            scenario, current_step_index, center_offset, do_rotate
                        )
                    
                    # trajs: [N, traj_len, 2]
                    if trajs.numel() == 0: continue

                    trajs_np = trajs[mask].cpu().numpy()  # [N_valid, traj_len, 2]
                    if trajs_np.size == 0: continue

                    # Flatten for clustering: [N_valid, traj_len*2]
                    new_vecs = trajs_np.reshape(trajs_np.shape[0], -1)
                    num_new = new_vecs.shape[0]

                    # -------- Reservoir sampling (Algorithm R) --------
                    if reservoir_count < max_vectors_total:
                        space = max_vectors_total - reservoir_count
                        take = min(num_new, space)
                        reservoir[reservoir_count : reservoir_count + take] = new_vecs[:take]
                        reservoir_count += take

                        if num_new > take:
                            leftover = new_vecs[take:]
                            start_idx = vectors_seen_total + take
                            indices = np.arange(start_idx, start_idx + len(leftover))
                            slots = rng.integers(0, indices + 1)
                            mask_keep = slots < max_vectors_total
                            if np.any(mask_keep):
                                reservoir[slots[mask_keep]] = leftover[mask_keep]
                    else:
                        # Reservoir full
                        indices_in_stream = np.arange(
                            vectors_seen_total, vectors_seen_total + num_new
                        )
                        candidate_slots = rng.integers(0, indices_in_stream + 1)
                        keep_mask = candidate_slots < max_vectors_total
                        if np.any(keep_mask):
                            dest_slots = candidate_slots[keep_mask]
                            reservoir[dest_slots] = new_vecs[keep_mask]

                    vectors_seen_total += num_new
                    scenarios_seen += 1

                    if scenarios_seen % 100 == 0:
                        gc.collect()
                        print(f"  Scanned {scenarios_seen} scenarios, seen {vectors_seen_total} trajectories...")

                    if max_scenarios and scenarios_seen >= max_scenarios:
                        break

                except Exception as e:
                    print(f"[RoadVectorTokenizer] Error: {e}")
                    continue
            
            del dataset
            gc.collect()

        if reservoir_count == 0:
            raise RuntimeError("No trajectories collected.")

        # Slice valid portion
        final_data = reservoir[:reservoir_count]  # [M, traj_len*2]
        print(f"[RoadVectorTokenizer] Final reservoir size: {final_data.shape}. Running k-means...")

        # K-means in flattened space
        centers_flat = self._kmeans(final_data, self.vocab_size, iters=kmeans_iters, rng=rng)
        
        # --- FIX: RESHAPE BACK TO 3D ---
        # centers_flat: [K, traj_len*2] -> [K, traj_len, 2]
        self.vocab_traj = centers_flat.reshape(self.vocab_size, self.traj_len, 2)
        self._vocab_flat = centers_flat

        print(
            f"[RoadVectorTokenizer] Built trajectory vocab: "
            f"{self.vocab_traj.shape} tokens (3D Shape Verified)"
        )

        self.save_smart_tokens("tokens/map_traj_token5.pkl")

    def save_smart_tokens(self, path: str) -> None:
        if self.vocab_traj is None:
            raise RuntimeError("Trajectory vocabulary not built yet.")

        traj_src = np.asarray(self.vocab_traj, dtype=np.float32)
        payload = {"traj_src": traj_src}

        with open(path, "wb") as f:
            pickle.dump(payload, f)

        print(f"[RoadVectorTokenizer] Saved SMART map tokens to {path}")

    # ----------------------------------------------------------------------
    # HELPERS
    # ----------------------------------------------------------------------
    def _ensure_scenario(self, item):
        if isinstance(item, scenario_pb2.Scenario): return item
        if isinstance(item, (bytes, bytearray)):
            s = scenario_pb2.Scenario(); s.ParseFromString(item); return s
        if isinstance(item, dict):
            # Handle various dict formats
            scen = item.get("scenario/bytes") or item.get("scenario")
            if hasattr(scen, "numpy"): scen = scen.numpy()
            if isinstance(scen, np.ndarray): scen = scen.item()
            s = scenario_pb2.Scenario(); s.ParseFromString(scen); return s
        raise TypeError(f"Unsupported scenario type: {type(item)}")

    def _extract_single_scene(self, scenario, current_step_index, center_offset, rotate):
        """
        Extracts 11-point polylines.
        Logic:
        1. Parse Polyline
        2. Subdivide long segments (>5m) (Optional but recommended for SMART fidelity)
        3. Resample to exactly 11 points.
        """
        heading = 0.0
        if center_offset is None:
            sdc_track = scenario.tracks[scenario.sdc_track_index]
            state = sdc_track.states[current_step_index]
            center_point = np.array([state.center_x, state.center_y], dtype=np.float32)
            heading = float(state.heading)
        else:
            center_point = np.array(center_offset, dtype=np.float32)

        if rotate:
            c = np.cos(-heading)
            s = np.sin(-heading)
            R = np.array([[c, -s], [s, c]], dtype=np.float32)
        else:
            R = np.eye(2, dtype=np.float32)

        all_trajs = []

        for feature in scenario.map_features:
            points = self._extract_polyline_points(feature)
            if points is None or len(points) < 2: continue

            pts = points.astype(np.float32)
            # Center & Rotate
            pts = pts - center_point[None, :]
            pts = pts @ R.T

            # Distance Filter
            if np.min(np.linalg.norm(pts, axis=1)) > self.max_dist:
                continue

            # 1. Subdivide (Break long lines) - Important for quality
            pts = self._subdivide_polyline(pts)

            # 2. Resample (To exactly traj_len points)
            traj = self._resample_polyline(pts, self.traj_len)  # [11, 2]
            
            if traj is not None:
                all_trajs.append(traj)

        if not all_trajs:
            trajs = th.zeros((self.max_vectors, self.traj_len, 2), dtype=th.float32)
            mask = th.zeros((self.max_vectors,), dtype=th.bool)
            return trajs, mask

        all_trajs = np.stack(all_trajs, axis=0) # [N, 11, 2]

        # Limit count
        if all_trajs.shape[0] > self.max_vectors:
            dists = np.linalg.norm(all_trajs[:, :, :2], axis=2).min(axis=1)
            idx = np.argsort(dists)[: self.max_vectors]
            all_trajs = all_trajs[idx]

        N = all_trajs.shape[0]
        mask = np.ones(N, dtype=bool)

        # Pad
        if N < self.max_vectors:
            pad = self.max_vectors - N
            pad_trajs = np.zeros((pad, self.traj_len, 2), dtype=np.float32)
            pad_mask = np.zeros(pad, dtype=bool)
            all_trajs = np.concatenate([all_trajs, pad_trajs], axis=0)
            mask = np.concatenate([mask, pad_mask], axis=0)

        return th.from_numpy(all_trajs).float(), th.from_numpy(mask).bool()

    def _extract_polyline_points(self, feature):
        if feature.HasField("lane"):
            return np.array([[p.x, p.y] for p in feature.lane.polyline], dtype=np.float32)
        if feature.HasField("road_line"):
            return np.array([[p.x, p.y] for p in feature.road_line.polyline], dtype=np.float32)
        if feature.HasField("road_edge"):
            return np.array([[p.x, p.y] for p in feature.road_edge.polyline], dtype=np.float32)
        if feature.HasField("crosswalk"):
            pts = np.array([[p.x, p.y] for p in feature.crosswalk.polygon], dtype=np.float32)
            if len(pts) > 0: pts = np.vstack([pts, pts[0]])
            return pts
        if feature.HasField("speed_bump"):
            pts = np.array([[p.x, p.y] for p in feature.speed_bump.polygon], dtype=np.float32)
            if len(pts) > 0: pts = np.vstack([pts, pts[0]])
            return pts
        if feature.HasField("stop_sign"):
            p = feature.stop_sign.position
            return np.array([[p.x, p.y], [p.x + 0.1, p.y]], dtype=np.float32)
        return None

    def _subdivide_polyline(self, points):
        """
        Splits segments longer than max_segment_len.
        """
        if len(points) < 2: return points
        
        new_points = [points[0]]
        for i in range(len(points) - 1):
            start = points[i]
            end = points[i+1]
            vec = end - start
            dist = np.linalg.norm(vec)
            
            if dist > self.max_segment_len:
                num_splits = int(np.ceil(dist / self.max_segment_len))
                step = vec / num_splits
                for k in range(1, num_splits + 1):
                    new_points.append(start + step * k)
            else:
                new_points.append(end)
                
        return np.array(new_points, dtype=np.float32)

    def _resample_polyline(self, points: np.ndarray, traj_len: int):
        if points.shape[0] < 2: return None

        diffs = np.diff(points, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        cumlen = np.concatenate([[0.0], np.cumsum(seg_lens)], axis=0)
        total_len = cumlen[-1]

        if total_len < 1e-3: return None

        s_samples = np.linspace(0.0, total_len, traj_len, dtype=np.float32)
        resampled = np.zeros((traj_len, 2), dtype=np.float32)
        
        # Vectorized Interpolation
        # Find indices where samples fall
        # np.searchsorted finds indices where elements should be inserted to maintain order
        # side='right' ensures we get the index of the segment *after* the sample
        idxs = np.searchsorted(cumlen, s_samples, side='right') - 1
        idxs = np.clip(idxs, 0, len(cumlen) - 2)
        
        s0 = cumlen[idxs]
        s1 = cumlen[idxs+1]
        
        # Alpha calculation with safe division
        denom = s1 - s0
        denom[denom < 1e-6] = 1.0 # Avoid div by zero
        alpha = (s_samples - s0) / denom
        
        p0 = points[idxs]
        p1 = points[idxs+1]
        
        resampled = (1.0 - alpha[:, None]) * p0 + alpha[:, None] * p1
        return resampled

    def _kmeans(self, X, k, iters=20, rng=None):
        if rng is None: rng = np.random.default_rng()
        M, D = X.shape
        if k > M: k = M

        centers = np.empty((k, D), dtype=np.float32)
        idx0 = rng.integers(0, M)
        centers[0] = X[idx0]

        closest_d2 = np.full(M, np.inf, dtype=np.float32)
        for c in range(1, k):
            diff = X - centers[c - 1]
            d2 = (diff ** 2).sum(axis=1)
            closest_d2 = np.minimum(closest_d2, d2)
            probs = closest_d2 / closest_d2.sum()
            idx = rng.choice(M, p=probs)
            centers[c] = X[idx]

        for _ in range(iters):
            diff = X[:, None, :] - centers[None, :, :]
            d2 = (diff ** 2).sum(axis=2)
            labels = d2.argmin(axis=1)
            for c in range(k):
                mask = (labels == c)
                if np.any(mask):
                    centers[c] = X[mask].mean(axis=0)
        return centers.astype(np.float32)


if __name__ == "__main__":
    import os
    import tensorflow as tf
    
    # Force CPU for TF
    tf.config.set_visible_devices([], 'GPU')

    DATASET_FOLDER = '/home/hansung/end2end_ad/datasets/waymo_open_dataset_motion_v_1_3_0/uncompressed/scenario/'
    TRAIN_FILES = os.path.join(DATASET_FOLDER, 'training/training.tfrecord*')
    
    filenames = tf.io.matching_files(TRAIN_FILES)
    filenames = tf.random.shuffle(filenames) 
    
    # Just convert to list of strings
    file_list = [f.numpy() for f in filenames]

    tokenizer = RoadVectorTokenizer(
        max_dist=200.0,
        max_vectors=2048,
        traj_len=11,
        vocab_size=1024,
        max_segment_len=5.0
    )

    tokenizer.build_vocabulary_from_scenarios(
        file_list,
        current_step_index=10,
        rotate=True,
        max_scenarios=5000,
        max_vectors_total=300_000,
        kmeans_iters=20,
        random_state=42,
    )