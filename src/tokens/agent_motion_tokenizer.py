from tokenizer import Tokenizer
import numpy as np
from waymo_open_dataset.protos import scenario_pb2
import matplotlib.pyplot as plt
import pickle
from typing import Dict, Optional, Tuple


SMART_AGENT_TYPE_KEY = {
    1: "veh",
    2: "ped",
    3: "cyc",
}


def _contours_from_traj(traj: np.ndarray,
                        length: float,
                        width: float) -> np.ndarray:
    """
    Convert an agent-centric trajectory [L,3] (x, y, yaw) into a sequence of
    polygon contours [L,4,2] using fixed length/width boxes.
    """
    x = traj[:, 0]
    y = traj[:, 1]
    theta = traj[:, 2]

    lf_x = x + 0.5 * length * np.cos(theta) - 0.5 * width * np.sin(theta)
    lf_y = y + 0.5 * length * np.sin(theta) + 0.5 * width * np.cos(theta)
    rf_x = x + 0.5 * length * np.cos(theta) + 0.5 * width * np.sin(theta)
    rf_y = y + 0.5 * length * np.sin(theta) - 0.5 * width * np.cos(theta)
    rb_x = x - 0.5 * length * np.cos(theta) + 0.5 * width * np.sin(theta)
    rb_y = y - 0.5 * length * np.sin(theta) - 0.5 * width * np.cos(theta)
    lb_x = x - 0.5 * length * np.cos(theta) - 0.5 * width * np.sin(theta)
    lb_y = y - 0.5 * length * np.sin(theta) + 0.5 * width * np.cos(theta)

    lf = np.stack([lf_x, lf_y], axis=-1)
    rf = np.stack([rf_x, rf_y], axis=-1)
    rb = np.stack([rb_x, rb_y], axis=-1)
    lb = np.stack([lb_x, lb_y], axis=-1)

    return np.stack([lf, rf, rb, lb], axis=1).astype(np.float32)


# Map Waymo Track types to tokenizer agent types
AGENT_TYPE_MAPPING = {
    scenario_pb2.Track.TYPE_UNSET: 0,
    scenario_pb2.Track.TYPE_OTHER: 0,
    scenario_pb2.Track.TYPE_VEHICLE: 1,     # vehicle
    scenario_pb2.Track.TYPE_PEDESTRIAN: 2,  # pedestrian
    scenario_pb2.Track.TYPE_CYCLIST: 3,     # cyclist/bicycle
}


class AgentMotionTokenizer(Tokenizer):
    """
    TrajTok-style tokenizer for 0.5s motion trajectories.

    Each token trajectory is:
      - length L (traj_len_steps) in time steps (e.g. L=5 at 10Hz → 0.5s)
      - agent-centric: the first state is at the origin (0,0)
      - first heading is aligned with +x (i.e., heading(0) = 0 rad)

    We build separate vocabularies per agent type (1=vehicle, 2=ped, 3=cyclist),
    then sparsify to target vocabulary sizes.
    """

    # Target vocabulary sizes per TrajTok paper
    TARGET_VOCAB_SIZE = {
        1: 8040,  # vehicles
        2: 3001,  # pedestrians
        3: 2798,  # cyclists
    }
    def __init__(self,
                 traj_len_steps: int = 5,
                 waymo_dt: float = 0.1,
                 random_state: int = 0,
                 max_scenarios: Optional[int] = None,
                 max_trajs_per_type: int = 200_000):
        """
        Args:
            traj_len_steps: L, number of 0.1s steps in each token trajectory.
            waymo_dt: time step of the raw Waymo trajectories (0.1s).
            random_state: seed for k-means in sparsification.
            max_scenarios: optional hard cap on scenarios to scan.
            max_trajs_per_type: max number of 0.5s windows to keep per agent type
                                (we use reservoir sampling to enforce this).
        """
        super().__init__()
        self.traj_len = traj_len_steps
        self.waymo_dt = waymo_dt
        self.rng = np.random.default_rng(random_state)

        self.max_scenarios = max_scenarios
        self.max_trajs_per_type = max_trajs_per_type


        # Per-type vocabulary: {agent_type: np.ndarray[num_tokens, L, 3] (x, y, yaw)}
        self.vocab: Dict[int, np.ndarray] = {}
        # Cached endpoints (x,y of final frame) per type: {agent_type: [K,2]}
        self.vocab_endpoints: Dict[int, np.ndarray] = {}

        # ---------------- TrajTok grid / filter parameters per type ----------------
        self.GRID_PARAMS = {
            1: {  # Vehicle
                "xmin": -5.0, "xmax": 20.0, "xinterval": 0.1,
                "ymin": -1.5, "ymax": 1.5, "yinterval": 0.05,
                "k": 4, "sp": 1, "sa": 20, "sr": 20,
            },
            3: {  # Bicycle / Cyclist
                "xmin": -1.0, "xmax": 8.0, "xinterval": 0.05,
                "ymin": -1.0, "ymax": 1.0, "yinterval": 0.05,
                "k": 4, "sp": 1, "sa": 20, "sr": 20,
            },
            2: {  # Pedestrian
                "xmin": -1.5, "xmax": 4.5, "xinterval": 0.05,
                "ymin": -2.0, "ymax": 2.0, "yinterval": 0.05,
                "k": 4, "sp": 1, "sa": 20, "sr": 20,
            },
        }

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def build_vocabulary_from_scenarios(self, scenarios):
        """
        Offline step: build vocabularies using all (or many) logged scenarios.

        Args:
            scenarios: iterable of Waymo `scenario_pb2.Scenario` protos,
                       raw serialized bytes, or tfds-style dicts.
        """
        print("[AgentMotionTokenizer] Starting vocabulary build from scenarios...")
        per_type_trajs = {1: [], 2: [], 3: []}
        # Total number of trajectories *seen* per type (before subsampling)
        per_type_counts = {1: 0, 2: 0, 3: 0}

        scenario_count = 0
        for item in scenarios:
            scenario_count += 1

            if self.max_scenarios is not None and scenario_count > self.max_scenarios:
                print(f"[AgentMotionTokenizer] Reached max_scenarios={self.max_scenarios}, stopping scan.")
                break

            if scenario_count % 500 == 0:
                print(f"[AgentMotionTokenizer] Processed {scenario_count} scenarios so far...")
                print("  Seen trajs so far:",
                      {t: per_type_counts[t] for t in (1, 2, 3)},
                      "  Kept in memory:",
                      {t: len(per_type_trajs[t]) for t in (1, 2, 3)})

            scenario = self._ensure_scenario(item)
            self._collect_trajs_from_scenario(scenario, per_type_trajs, per_type_counts)

            # If we've filled all type buckets, we can optionally stop early
            if (self.max_trajs_per_type is not None
                and all(per_type_counts[t] >= self.max_trajs_per_type for t in (1, 2, 3))):
                print("[AgentMotionTokenizer] Reached max_trajs_per_type for all types. Stopping scan.")
                break

        print(f"[AgentMotionTokenizer] Finished scanning {scenario_count} scenarios.")
        for t in (1, 2, 3):
            print(f"[AgentMotionTokenizer] Seen total {per_type_counts[t]} trajs for type {t}, "
                  f"kept {len(per_type_trajs[t])} in memory.")

        # Per-type vocab building + sparsification
        for agent_type, trajs in per_type_trajs.items():
            if len(trajs) == 0:
                print(f"[AgentMotionTokenizer] No trajectories for agent_type={agent_type}, skipping.")
                continue

            print(f"[AgentMotionTokenizer] Building dense vocab for agent_type={agent_type}...")
            params = self.GRID_PARAMS[agent_type]
            D = np.asarray(trajs, dtype=np.float32)

            dense_vocab = self._build_vocab_for_type(D, params)
            print(f"[AgentMotionTokenizer] Dense vocab for type {agent_type} has {dense_vocab.shape[0]} tokens.")

            target_K = self.TARGET_VOCAB_SIZE.get(agent_type, None)
            if target_K is not None and dense_vocab.shape[0] > target_K:
                print(f"[AgentMotionTokenizer] Sparsifying type {agent_type} "
                      f"{dense_vocab.shape[0]} → {target_K} via k-means on endpoints...")
                vocab = self._sparsify_vocab(dense_vocab, target_K)
                print(f"[AgentMotionTokenizer] Done sparsifying type {agent_type}: "
                      f"{dense_vocab.shape[0]} → {vocab.shape[0]}")
            else:
                vocab = dense_vocab
                print(f"[AgentMotionTokenizer] No sparsification needed for type {agent_type}.")

            self.vocab[agent_type] = vocab
            self.vocab_endpoints[agent_type] = vocab[:, -1, 0:2]

            print(f"[AgentMotionTokenizer] Final vocab type {agent_type} with {vocab.shape[0]} tokens.")

        print("[AgentMotionTokenizer] Vocabulary build complete for all agent types.")


    def encode(self, traj_local: np.ndarray, agent_type: int):
        """
        Quantize a single 0.5s trajectory to the nearest token.

        Args:
            traj_local: np.ndarray [L, 3] in agent-centric coords (x, y, yaw),
                        where the first heading is ~0 rad (aligned with +x).
            agent_type: mapped agent type (1=vehicle, 2=ped, 3=cyclist).

        Returns:
            token_idx: int index into self.vocab[agent_type]
            dist:      float average squared error to that token
        """
        if agent_type not in self.vocab:
            raise ValueError(f"No vocabulary built for agent_type={agent_type}.")

        V = self.vocab[agent_type]  # [K, L, 3]
        if traj_local.shape != V.shape[1:]:
            raise ValueError(f"Trajectory shape {traj_local.shape} does not match token shape {V.shape[1:]}")

        diff = V - traj_local[None, :, :]
        diff_xy = diff[..., 0:2]
        yaw_diff = diff[..., 2]
        yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))

        sq = (diff_xy ** 2).sum(axis=-1) + yaw_diff ** 2
        err_per_token = sq.mean(axis=-1)

        idx = int(err_per_token.argmin())
        return idx, float(err_per_token[idx])

    def decode_token(self, token_idx: int, agent_type: int) -> np.ndarray:
        """Return the trajectory for a given token (np.ndarray [L, 3])."""
        return self.vocab[agent_type][token_idx]

    # ------------------------------------------------------------------
    # SMART-compatible export helpers
    # ------------------------------------------------------------------

    def export_smart_tokens(self,
                            dims: Optional[Dict[int, Tuple[float, float]]] = None
                            ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Convert TrajTok vocab to the SMART TokenProcessor format:

          token:     (tokens, 4, 2)
          traj:      (tokens, 6, 3)     # x, y, heading
          token_all: (tokens, 6, 4, 2)  # 6 frames, 4 corners, x/y

        We always use the first 6 frames of each trajectory. If the stored
        TrajTok vocab has L < 6, we pad by repeating the last frame.
        """
        if not self.vocab:
            raise RuntimeError("No vocabulary to export. Run build_vocabulary_from_scenarios first.")

        print("[AgentMotionTokenizer] Exporting SMART-format tokens...")

        # Default length/width per agent type
        default_dims = {
            1: (4.8, 2.0),  # vehicle
            3: (2.0, 1.0),  # cyclist
            2: (1.0, 1.0),  # pedestrian
        }
        dims = dims or default_dims

        token: Dict[str, np.ndarray] = {}
        traj: Dict[str, np.ndarray] = {}
        token_all: Dict[str, np.ndarray] = {}

        NUM_FRAMES = 6  # SMART spec: use first 6 frames

        for agent_type, trajs in self.vocab.items():
            if trajs is None or len(trajs) == 0:
                print(f"[AgentMotionTokenizer] export_smart_tokens: no tokens for type {agent_type}, skipping.")
                continue

            key = SMART_AGENT_TYPE_KEY.get(agent_type)
            if key is None:
                print(f"[AgentMotionTokenizer] export_smart_tokens: unknown SMART key for type {agent_type}, skipping.")
                continue

            length, width = dims.get(agent_type, default_dims[1])

            # trajs: [K, L, 3]  (x, y, yaw)
            trajs = np.asarray(trajs, dtype=np.float32)
            K, L, D = trajs.shape
            assert D == 3, f"Expected traj dim 3 (x,y,yaw), got {D}"

            print(f"[AgentMotionTokenizer] Exporting type {agent_type} ({key}): K={K}, L={L}")

            # Build 6-frame trajectories: [K, 6, 3]
            traj6 = np.zeros((K, NUM_FRAMES, 3), dtype=np.float32)

            if L >= NUM_FRAMES:
                traj6[:, :, :] = trajs[:, :NUM_FRAMES, :]
            else:
                traj6[:, :L, :] = trajs
                traj6[:, L:, :] = trajs[:, -1:, :]

            # Contours from these 6-frame trajectories: [K, 6, 4, 2]
            contours6 = np.stack(
                [_contours_from_traj(traj6[k], length, width) for k in range(K)],
                axis=0
            ).astype(np.float32)

            token[key] = contours6[:, 0]        # first frame's box: [K, 4, 2]
            traj[key] = traj6                   # [K, 6, 3]
            token_all[key] = contours6          # [K, 6, 4, 2]

        print("[AgentMotionTokenizer] SMART-format export complete.")
        return {"token": token, "traj": traj, "token_all": token_all}

    def save_smart_tokens(self,
                          path: str,
                          dims: Optional[Dict[int, Tuple[float, float]]] = None) -> None:
        """Save SMART-format pickle usable by preprocess/tokenizer.py."""
        print(f"[AgentMotionTokenizer] Saving SMART tokens to {path}...")
        payload = self.export_smart_tokens(dims=dims)
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print(f"[AgentMotionTokenizer] Saved SMART tokens to {path}.")

    # ------------------------------------------------------------------
    # VISUALIZATION (like Figure 1b)
    # ------------------------------------------------------------------

    def visualize_vocabulary(self,
                             agent_type: int = 1,
                             show_full_trajectory: bool = False,
                             arrow_scale: float = 1.0,
                             figsize=(6, 6),
                             save_path: str | None = None):
        """
        Visualize the vocabulary for a given agent type as arrows, similar
        to Fig. 1b in SMART.

        Args:
            agent_type: 1=vehicle, 2=pedestrian, 3=cyclist.
            show_full_trajectory: if True, plot all samples along each
                                  token trajectory; otherwise, only an arrow
                                  from origin to endpoint.
            arrow_scale: scalar to rescale arrow lengths for nicer plotting.
            figsize: matplotlib figure size.
            save_path: if not None, save figure to this path.
        """

        if agent_type not in self.vocab:
            raise ValueError(f"No vocabulary for agent_type={agent_type}.")

        V = self.vocab[agent_type]  # [K, L, 3]
        K, L, _ = V.shape

        print(f"[AgentMotionTokenizer] Visualizing vocab for agent_type={agent_type} with K={K} tokens...")

        fig, ax = plt.subplots(figsize=figsize)

        for k in range(K):
            traj = V[k]
            xs = traj[:, 0]
            ys = traj[:, 1]

            if show_full_trajectory:
                ax.plot(xs, ys, linewidth=0.5)
                dx = (xs[-1] - xs[-2]) * arrow_scale
                dy = (ys[-1] - ys[-2]) * arrow_scale
                ax.arrow(xs[-2], ys[-2], dx, dy,
                         head_width=0.05, head_length=0.1, length_includes_head=True)
            else:
                dx = xs[-1] * arrow_scale
                dy = ys[-1] * arrow_scale
                ax.arrow(0.0, 0.0, dx, dy,
                         head_width=0.05, head_length=0.1, length_includes_head=True)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (agent-centric, m)")
        ax.set_ylabel("y (agent-centric, m)")
        ax.set_title(f"AgentMotion Vocab (type={agent_type}, K={K})")
        ax.grid(True)

        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight", dpi=200)
            print(f"[AgentMotionTokenizer] Saved vocabulary visualization to {save_path}.")
        plt.show()

    # ------------------------------------------------------------------
    # INTERNAL: scenario handling + agent-centric rotation
    # ------------------------------------------------------------------

    def _ensure_scenario(self, item):
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
            if hasattr(scen_bytes, "numpy"):
                scen_bytes = scen_bytes.numpy()
            if isinstance(scen_bytes, np.ndarray):
                scen_bytes = scen_bytes.item()
            s = scenario_pb2.Scenario()
            s.ParseFromString(scen_bytes)
            return s

        raise TypeError(f"Unsupported scenario item type: {type(item)}")
   
    def _collect_trajs_from_scenario(self, scenario, per_type_trajs, per_type_counts):
        """
        Extract all length-L agent-centric windows from a scenario and feed them
        into per-type reservoirs.
        """
        L = self.traj_len

        for track in scenario.tracks:
            agent_type = AGENT_TYPE_MAPPING.get(track.object_type, 0)
            if agent_type == 0:
                continue

            states = track.states
            T = len(states)
            if T < L:
                continue

            for t0 in range(0, T - L + 1):
                window = states[t0:t0 + L]
                if not all(st.valid for st in window):
                    continue

                traj_local = self._make_agent_centric_traj(window)
                self._reservoir_add_trajectory(agent_type,
                                               traj_local,
                                               per_type_trajs,
                                               per_type_counts)

    def _reservoir_add_trajectory(self,
                                  agent_type: int,
                                  traj_local: np.ndarray,
                                  per_type_trajs: Dict[int, list],
                                  per_type_counts: Dict[int, int]) -> None:
        """
        Reservoir sampling for trajectories per agent_type.

        - per_type_counts[agent_type]: total number of trajs SEEN so far.
        - per_type_trajs[agent_type]: list of KEPT trajs, capped at max_trajs_per_type.
        """
        c = per_type_counts[agent_type]
        per_type_counts[agent_type] = c + 1

        limit = self.max_trajs_per_type
        if limit is None:
            # Unlimited: just append (not recommended for memory)
            per_type_trajs[agent_type].append(traj_local)
            return

        # If we haven't filled the reservoir yet, just append
        if len(per_type_trajs[agent_type]) < limit:
            per_type_trajs[agent_type].append(traj_local)
            return

        # Reservoir sampling: replace existing item with prob = limit / (c+1)
        j = self.rng.integers(0, c + 1)
        if j < limit:
            per_type_trajs[agent_type][j] = traj_local

    def _make_agent_centric_traj(self, window):
        L = len(window)
        origin = window[0]

        x0 = origin.center_x
        y0 = origin.center_y
        yaw0 = origin.heading

        c = np.cos(-yaw0)
        s = np.sin(-yaw0)
        R = np.array([[c, -s],
                      [s,  c]], dtype=np.float32)

        traj = np.zeros((L, 3), dtype=np.float32)

        for k, st in enumerate(window):
            pos = np.array([st.center_x - x0,
                            st.center_y - y0], dtype=np.float32)
            pos_local = pos @ R

            yaw_rel = st.heading - yaw0
            yaw_rel = (yaw_rel + np.pi) % (2 * np.pi) - np.pi

            traj[k, 0:2] = pos_local
            traj[k, 2] = yaw_rel

        return traj

    # ------------------------------------------------------------------
    # VOCAB BUILDING FOR ONE TYPE (TrajTok logic)
    # ------------------------------------------------------------------

    def _build_vocab_for_type(self, D: np.ndarray, params: dict) -> np.ndarray:
        """
        Build dense TrajTok vocabulary for one agent type (no size control yet).

        Args:
            D: np.ndarray [N, L, 3] (agent-centric trajectories)
            params: grid + filter hyper-params.

        Returns:
            dense_vocab: np.ndarray [K_full, L, 3]
        """
        print(f"[AgentMotionTokenizer] _build_vocab_for_type: starting with {D.shape[0]} trajectories, "
              f"L={D.shape[1]}")
        # 1) Flip trajectories in y / yaw to enforce left-right symmetry
        D_flipped = self._flip_trajectories(D)
        De = np.concatenate([D, D_flipped], axis=0)
        print(f"[AgentMotionTokenizer] _build_vocab_for_type: after flip/aug, {De.shape[0]} trajectories.")

        # 2) Use endpoints (x_L, y_L) for gridding
        endpoints = De[:, -1, 0:2]

        (cell_indices,
         Ntraj,
         B_init) = self._grid_and_preselect(endpoints, params)
        print("[AgentMotionTokenizer] _build_vocab_for_type: gridding complete.")

        # 3) Filter and expand active cells
        B_hat = self._filter_and_expand(B_init, params)
        num_active = int(B_hat.sum())
        print(f"[AgentMotionTokenizer] _build_vocab_for_type: {num_active} active cells after filtering/expansion.")

        # 4) Generate one token per active cell
        dense_tokens = self._generate_tokens_from_grid(
            De, cell_indices, Ntraj, B_hat, params
        )
        print(f"[AgentMotionTokenizer] _build_vocab_for_type: generated {len(dense_tokens)} dense tokens.")

        return np.asarray(dense_tokens, dtype=np.float32)

    # ------------------- helpers: Step 1 -------------------

    def _flip_trajectories(self, D: np.ndarray) -> np.ndarray:
        """
        Flip trajectories across the x-axis (y → -y, yaw → -yaw) to exploit
        left-right symmetry.
        """
        D_flipped = D.copy()
        D_flipped[..., 1] = -D_flipped[..., 1]
        D_flipped[..., 2] = -D_flipped[..., 2]
        D_flipped[..., 2] = (D_flipped[..., 2] + np.pi) % (2 * np.pi) - np.pi
        return D_flipped

    # ------------------- helpers: Step 2 -------------------

    def _grid_and_preselect(self, endpoints: np.ndarray, params: dict):
        xmin = params["xmin"]
        xmax = params["xmax"]
        ymin = params["ymin"]
        ymax = params["ymax"]
        dx = params["xinterval"]
        dy = params["yinterval"]
        sp = params["sp"]

        Nx = int(np.floor((xmax - xmin) / dx))
        Ny = int(np.floor((ymax - ymin) / dy))

        cell_indices = [[[] for _ in range(Ny)] for _ in range(Nx)]
        Ntraj = np.zeros((Nx, Ny), dtype=np.int32)

        for idx, (x, y) in enumerate(endpoints):
            if not (xmin <= x < xmax and ymin <= y < ymax):
                continue
            i = int((x - xmin) // dx)
            j = int((y - ymin) // dy)
            if 0 <= i < Nx and 0 <= j < Ny:
                cell_indices[i][j].append(idx)
                Ntraj[i, j] += 1

        B_init = (Ntraj >= sp).astype(np.uint8)
        return cell_indices, Ntraj, B_init

    # ------------------- helpers: Step 3 -------------------

    def _filter_and_expand(self, B: np.ndarray, params: dict) -> np.ndarray:
        """
        Apply neighborhood-based filter/expansion to get final active cells B_hat.
        """
        k = params["k"]
        sa = params["sa"]
        sr = params["sr"]

        Nx, Ny = B.shape
        Nvb = np.zeros_like(B, dtype=np.int32)

        for i in range(Nx):
            i0 = max(0, i - k)
            i1 = min(Nx - 1, i + k)
            for j in range(Ny):
                j0 = max(0, j - k)
                j1 = min(Ny - 1, j + k)
                Nvb[i, j] = B[i0:i1 + 1, j0:j1 + 1].sum()

        B_hat = B.copy()
        for i in range(Nx):
            for j in range(Ny):
                if B[i, j] == 0 and Nvb[i, j] >= sa:
                    B_hat[i, j] = 1
                elif B[i, j] == 1 and Nvb[i, j] <= sr:
                    B_hat[i, j] = 0
        return B_hat

    # ------------------- helpers: Step 4 -------------------

    def _generate_tokens_from_grid(self,
                                   De: np.ndarray,
                                   cell_indices,
                                   Ntraj: np.ndarray,
                                   B_hat: np.ndarray,
                                   params: dict):
        xmin = params["xmin"]
        ymin = params["ymin"]
        dx = params["xinterval"]
        dy = params["yinterval"]
        k = params["k"]

        Nx, Ny = B_hat.shape
        tokens = []

        for i in range(Nx):
            for j in range(Ny):
                if B_hat[i, j] == 0:
                    continue

                idxs = cell_indices[i][j]
                if len(idxs) > 0:
                    trajs_ij = De[np.asarray(idxs, dtype=np.int64)]
                    token = trajs_ij.mean(axis=0)
                else:
                    px = xmin + (i + 0.5) * dx
                    py = ymin + (j + 0.5) * dy
                    p_end = np.array([px, py], dtype=np.float32)

                    yaw_end = self._estimate_endpoint_yaw(
                        De, cell_indices, i, j, k
                    )
                    token = self._curve_interp_to_endpoint(
                        p_end, yaw_end, De.shape[1]
                    )
                tokens.append(token)

        return tokens

    def _estimate_endpoint_yaw(self,
                               De: np.ndarray,
                               cell_indices,
                               i: int,
                               j: int,
                               k: int) -> float:
        Nx = len(cell_indices)
        Ny = len(cell_indices[0])

        yaws = []
        i0 = max(0, i - k)
        i1 = min(Nx - 1, i + k)
        j0 = max(0, j - k)
        j1 = min(Ny - 1, j + k)

        for ii in range(i0, i1 + 1):
            for jj in range(j0, j1 + 1):
                idxs = cell_indices[ii][jj]
                for idx in idxs:
                    yaws.append(De[idx, -1, 2])

        if len(yaws) == 0:
            return 0.0

        yaws = np.asarray(yaws, dtype=np.float32)
        s = np.sin(yaws).mean()
        c = np.cos(yaws).mean()
        return float(np.arctan2(s, c))

    def _curve_interp_to_endpoint(self,
                                  p_end: np.ndarray,
                                  yaw_end: float,
                                  L: int) -> np.ndarray:
        """
        Simple linear interpolation from origin to p_end with constant yaw_end.
        """
        token = np.zeros((L, 3), dtype=np.float32)
        if L == 1:
            token[0, 0:2] = p_end
            token[0, 2] = yaw_end
            return token

        for k in range(L):
            t = k / (L - 1)
            token[k, 0:2] = p_end * t
            token[k, 2] = yaw_end
        return token

    # ------------------------------------------------------------------
    # SPARSIFICATION: dense → target_K tokens (per type)
    # ------------------------------------------------------------------

    def _sparsify_vocab(self, tokens: np.ndarray, target_K: int) -> np.ndarray:
        """
        Compress dense tokens to target_K by clustering endpoints and averaging
        trajectories within each cluster.

        Args:
            tokens: [K_full, L, 3]
            target_K: desired number of tokens

        Returns:
            new_tokens: [target_K, L, 3]
        """
        K_full, L, D = tokens.shape
        if target_K >= K_full:
            return tokens

        print(f"[AgentMotionTokenizer] _sparsify_vocab: K_full={K_full}, target_K={target_K}")

        # Use endpoints as clustering features
        E = tokens[:, -1, 0:2]  # [K_full, 2]
        centers, labels = self._kmeans_2d(E, target_K)

        new_tokens = np.zeros((target_K, L, D), dtype=np.float32)

        for k in range(target_K):
            mask = (labels == k)
            if np.any(mask):
                new_tokens[k] = tokens[mask].mean(axis=0)
            else:
                d2 = ((E - centers[k]) ** 2).sum(axis=1)
                idx = int(d2.argmin())
                new_tokens[k] = tokens[idx]

        return new_tokens

    def _kmeans_2d(self, X: np.ndarray, k: int, iters: int = 20):
        """
        Simple k-means on 2D points X [N,2] with k-means++ init.

        Returns:
            centers: [k,2]
            labels:  [N]
        """
        rng = self.rng
        N, D = X.shape
        if k > N:
            k = N

        centers = np.empty((k, D), dtype=np.float32)

        # k-means++ init
        idx0 = rng.integers(0, N)
        centers[0] = X[idx0]

        closest_d2 = np.full(N, np.inf, dtype=np.float32)
        for c in range(1, k):
            diff = X - centers[c - 1]
            d2 = (diff ** 2).sum(axis=1)
            closest_d2 = np.minimum(closest_d2, d2)
            probs = closest_d2 / (closest_d2.sum() + 1e-12)
            idx = rng.choice(N, p=probs)
            centers[c] = X[idx]

        for it in range(iters):
            diff = X[:, None, :] - centers[None, :, :]
            d2 = (diff ** 2).sum(axis=2)
            labels = d2.argmin(axis=1)

            for c in range(k):
                mask = (labels == c)
                if np.any(mask):
                    centers[c] = X[mask].mean(axis=0)

            if (it + 1) % 5 == 0:
                print(f"[AgentMotionTokenizer] _kmeans_2d: iteration {it+1}/{iters} complete.")

        return centers, labels


if __name__ == "__main__":
    import os
    from waymo_open_dataset.protos import scenario_pb2
    import tensorflow as tf

    DATASET_FOLDER = '/home/hansung/end2end_ad/datasets/waymo_open_dataset_motion_v_1_3_0/uncompressed/scenario/'
    TRAIN_FILES = os.path.join(DATASET_FOLDER, 'training/training.tfrecord*')
    filenames = tf.io.matching_files(TRAIN_FILES)
    dataset = tf.data.TFRecordDataset(filenames)
    dataset_iterator = dataset.as_numpy_iterator()

    print("[AgentMotionTokenizer] Main: creating tokenizer...")
    tok = AgentMotionTokenizer(
        traj_len_steps=5,
        waymo_dt=0.1,
        random_state=0,
        max_scenarios=50_000,        # or None if you just want to rely on traj cap
        max_trajs_per_type=500_000,  
    )

    print("[AgentMotionTokenizer] Main: building vocabulary from training scenarios...")
    tok.build_vocabulary_from_scenarios(dataset_iterator)

    # SMART-format export for preprocess/tokenizer.py
    out_path = "tokens/cluster_frame_5_2048.pkl"
    tok.save_smart_tokens(out_path)
    print(f"[AgentMotionTokenizer] Main: SMART tokens saved to {out_path}.")

    # 3) Visualize
    tok.visualize_vocabulary(agent_type=1, show_full_trajectory=False)  # vehicles
    tok.visualize_vocabulary(agent_type=2)  # pedestrians
    tok.visualize_vocabulary(agent_type=3)  # cyclists
