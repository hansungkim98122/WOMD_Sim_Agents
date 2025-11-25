import torch as th
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tqdm import tqdm
from waymo_open_dataset.protos import scenario_pb2
from agent_motion_tokenizer import AgentMotionTokenizer


if __name__ == '__main__':
    # Vehicles (type 1) – arrow from origin to endpoint

    tok = AgentMotionTokenizer(traj_len_steps=5, waymo_dt=0.1)

    def load_agent_vocab_from_npz(tokenizer: AgentMotionTokenizer, path: str):
        data = np.load(path, allow_pickle=True)

        tokenizer.vocab = {}
        tokenizer.vocab_endpoints = {}

        # Only add types that exist in the file

        for agent_type, key in [(1, "vehicle"), (2, "pedestrian"), (3, "cyclist")]:
            if key in data:
                vocab_arr = data[key]
                if vocab_arr.size == 0:
                    continue
                tokenizer.vocab[agent_type] = vocab_arr
                tokenizer.vocab_endpoints[agent_type] = vocab_arr[:, -1, 0:2]

    # usage:
    load_agent_vocab_from_npz(tok, "../token_vocabs/trajtok_vocab.npz")

    tok.visualize_vocabulary(agent_type=1,
                            show_full_trajectory=False,
                            arrow_scale=1.0,
                            figsize=(5, 5),
                            save_path="vocab_vehicle_arrows.png")

    # Pedestrians (type 2) – full polyline + little arrow at the tail
    tok.visualize_vocabulary(agent_type=2,
                            show_full_trajectory=True,
                            arrow_scale=1.0,
                            figsize=(5, 5))
