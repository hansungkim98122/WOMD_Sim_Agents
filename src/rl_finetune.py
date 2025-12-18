import os
import math
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
from argparse import ArgumentParser

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch_geometric.data import Batch

# Waymo TF metric (be careful: TF runs on CPU-only here)
from waymo_open_dataset.wdl_limited.sim_agents_metrics.interaction_features import (
    compute_distance_to_nearest_object,
)

from utils.config import load_config_act
from datamodule.datamodule import MultiDataModule
from model.smart import SMART

COLLISION_DISTANCE_THRESHOLD = 0.0


@torch.no_grad()
def reward2go(reward, valid, eval_m, B, N, T, gamma=0.95, device="cuda"):
    """
    reward: (B, N, T) float
    valid:  (B, S) bool (S >= T)
    eval_m: (B,) bool
    """
    returns = torch.zeros_like(reward)
    running = torch.zeros((B, N), device=device)

    for t in reversed(range(T)):
        running = reward[:, :, t] + gamma * running
        returns[:, :, t] = running

    mask = (valid[:, :T] & eval_m[:, None]).float().to(device)  # (B,T)
    mask = mask[:, None, :]  # (B,1,T)

    denom = mask.sum().clamp_min(1.0)
    mean = (returns * mask).sum() / denom
    var = (((returns - mean) ** 2) * mask).sum() / denom
    std = torch.sqrt(var + 1e-8)
    norm_returns = (returns - mean) / (std + 1e-8)
    return norm_returns


class RLFineTuneModule(pl.LightningModule):
    def __init__(self, cfg, base_model: SMART):
        super().__init__()
        self.save_hyperparameters(ignore=["base_model"])

        self.cfg = cfg
        self.model = base_model

        self.T_pred = cfg.time_info.num_future_steps
        self.gamma = cfg.finetuning.gamma
        self.lambda_collision = cfg.finetuning.collision
        self.lr = float(cfg.finetuning.lr)

        # fine-tune only motion network (as you wrote)
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in self.model.encoder.agent_encoder.parameters():
            p.requires_grad_(True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.model.encoder.agent_encoder.parameters(),
            lr=self.lr,
        )
        return opt

    def _mem(self, tag):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            a = torch.cuda.memory_allocated() / 1e9
            r = torch.cuda.memory_reserved() / 1e9
            p = torch.cuda.max_memory_allocated() / 1e9
            print(f"[{tag}] alloc={a:.2f}G reserved={r:.2f}G peak={p:.2f}G")

    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            torch.cuda.reset_peak_memory_stats()
        self._mem(f"start {batch_idx}")

        device = self.device

        batch = self.model.match_token_map(batch)
        batch = self.model.sample_pt_pred(batch)
        if isinstance(batch, Batch):
            batch["agent"]["av_index"] += batch["agent"]["ptr"][:-1]
        # One dataloader item = one scene/graph (typical for your setup)
        bs = int(getattr(batch, "num_graphs", 1))

        state = self.model.inference_one_step_init(batch)
        B = state["num_agent"]
        T = state["num_token_steps"]
        shift = self.model.encoder.agent_encoder.shift

        first_future = self.model.encoder.agent_encoder.num_historical_steps

        gt = batch["agent"]["position"][
            :, first_future:, : self.model.encoder.agent_encoder.input_dim
        ].contiguous()  # (B, future_steps, 2)

        valid_mask_womd = batch["agent"]["valid_mask"].to(device)  # (B, steps)
        # If state["agent_valid_mask"] is token-time mask, you can still use womd for step-time validity:
        # valid_mask_full = state["agent_valid_mask"].to(device)

        logp_list = []
        r_list = []
        evalm_list = []
        coll_list = []

        for t in range(T):
            # per-step eval mask (B,)
            step_idx = first_future + t * shift
            eval_mask_t = valid_mask_womd[:, step_idx]  # bool (B,)
            evalm_list.append(eval_mask_t)

            out = self.model.inference_rollout_step(state, t)

            action_logp = out["action_logp"]
            assert action_logp.requires_grad, "action_logp must require grad"
            logp_list.append(action_logp)  # (B,)

            pred_center = state["pos_a"]  # (B, token_time, 2)
            hist_token_idx = state["hist_token_idx"]

            # collision penalty: compute distance using ALL objects, evaluate only eval_mask_t
            with torch.no_grad():
                # tracking reward aligned to gt[:, t*shift]
                # gt index in gt-space is (t*shift)
                gt_idx = t * shift
                if gt_idx >= gt.shape[1]:
                    # safety if T*shift > gt horizon
                    break

                track_err = torch.linalg.norm(
                    pred_center[:, hist_token_idx + t, :] - gt[:, gt_idx, :],
                    dim=-1,
                )  # (B,)
                r = -track_err

                # center_x/y for ALL objects, one timestep -> shape (B,1)
                cx = pred_center[:, hist_token_idx + t, 0].detach().cpu().numpy()[:, None]
                cy = pred_center[:, hist_token_idx + t, 1].detach().cpu().numpy()[:, None]
                cz = np.zeros_like(cx, dtype=np.float32)  # you don't predict z; ok for 2D

                shp = batch["agent"]["shape"].detach().cpu().numpy()  # (B, steps, 3)
                tt = min(step_idx, shp.shape[1] - 1)
                L = shp[:, tt, 0:1]
                W = shp[:, tt, 1:2]
                H = shp[:, tt, 2:3] if shp.shape[-1] > 2 else np.ones_like(L)

                yaw = state["head_a"][:, hist_token_idx + t].detach().cpu().numpy()[:, None]

                # TF fn returns (num_eval_objects, num_steps) AFTER gathering by evaluated_object_mask.
                # So set evaluated_object_mask with shape (B,)
                dist_tf = compute_distance_to_nearest_object(
                    center_x=cx,
                    center_y=cy,
                    center_z=cz,
                    length=L,
                    width=W,
                    height=H,
                    heading=yaw,
                    valid=np.ones((B, 1), dtype=bool),
                    evaluated_object_mask=eval_mask_t.detach().cpu().numpy().astype(bool),
                    corner_rounding_factor=0.1,
                )  # (num_eval, 1)

                # map dist back into (B,)
                col_full = torch.zeros(B, device=device)
                if dist_tf.shape[0] > 0:
                    dist_eval = dist_tf.numpy()[:, 0]
                    col_eval = (dist_eval <= COLLISION_DISTANCE_THRESHOLD).astype(np.float32)
                    col_full[eval_mask_t] = torch.from_numpy(col_eval).to(device)

            r = r - self.lambda_collision * col_full

            # apply eval mask to reward (and optionally validity)
            r = r * eval_mask_t.float()

            r_list.append(r)                 # (B,)
            coll_list.append(col_full)       # (B,)

        # stack: (B,T)
        logp = torch.stack(logp_list, dim=1)
        rewards = torch.stack(r_list, dim=1)
        evalm = torch.stack(evalm_list, dim=1).float()
        colls = torch.stack(coll_list, dim=1)

        # returns-to-go (B,T)
        # (no need for eval_m inside reward2go; just mask outside)
        with torch.no_grad():
            returns = torch.zeros_like(rewards)
            running = torch.zeros(B, device=device)
            for t in reversed(range(rewards.shape[1])):
                running = rewards[:, t] + self.gamma * running
                returns[:, t] = running

            # normalize over masked entries
            denom = evalm.sum().clamp_min(1.0)
            mean = (returns * evalm).sum() / denom
            var = (((returns - mean) ** 2) * evalm).sum() / denom
            adv = (returns - mean) / (torch.sqrt(var + 1e-8) + 1e-8)

        # REINFORCE loss (automatic optimization)
        loss = -((logp * adv) * evalm).sum() / evalm.sum().clamp_min(1.0)

        self.log("loss", loss.detach(), on_step=True, on_epoch=True, prog_bar=True,
                batch_size=bs, sync_dist=True)

        self.log("avg_r", ((rewards * evalm).sum() / evalm.sum().clamp_min(1.0)).detach(),
                on_step=True, on_epoch=True, prog_bar=True,
                batch_size=bs, sync_dist=True)

        self.log("avg_coll", ((colls * evalm).sum() / evalm.sum().clamp_min(1.0)).detach(),
                on_step=True, on_epoch=True, prog_bar=True,
                batch_size=bs, sync_dist=True)

        self._mem(f"end {batch_idx}")

        return loss



def main():
    parser = ArgumentParser()
    Predictor_hash = {"smart": SMART}

    parser.add_argument("--config", type=str, default="configs/training/training_rl.yaml")
    parser.add_argument("--log_dir", type=str, default="tb_logs")
    args = parser.parse_args()

    cfg = load_config_act(args.config)

    Predictor = Predictor_hash[cfg.Model.predictor]

    # Data
    Data_config = cfg.Dataset
    Data_config["smart_token"] = cfg.Model.use_smart_tokens
    datamodule = MultiDataModule(**vars(Data_config))

    # Base model
    if cfg.finetuning.pretrain_ckpt:
        print(f"Loading pretrained model from: {cfg.finetuning.pretrain_ckpt}")
        base_model = Predictor.load_from_checkpoint(cfg.finetuning.pretrain_ckpt, config=cfg.Model)
    else:
        base_model = Predictor(cfg.Model)

    # Lightning RL wrapper
    rl_module = RLFineTuneModule(cfg, base_model)

    # Logger
    tb_logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name="smart_rl",
    )

    # Checkpoints: monitor avg reward (maximize). Also save_last.
    ckpt_cb = ModelCheckpoint(
        dirpath=cfg.finetuning.save_ckpt_path,
        filename="rl-{epoch:02d}-{step:06d}-{avg_r:.4f}",
        monitor="avg_r",
        mode="max",
        save_top_k=5,
        save_last=True,
        every_n_epochs=1,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer_cfg = cfg.Trainer
    use_cuda = torch.cuda.is_available()

    # If your yaml says accelerator="gpu" but you run on a machine without CUDA,
    # Lightning will throw exactly this MisconfigurationException.
    if use_cuda:
        accelerator = "gpu"
        devices = 1 if getattr(cfg.Trainer, "devices", None) in (None, "auto") else cfg.Trainer.devices
    else:
        accelerator = "cpu"
        devices = 1

    print(f"[RL] torch.cuda.is_available()={use_cuda} -> accelerator={accelerator}, devices={devices}")

    strategy = DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True)

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        num_nodes=1,
        max_epochs=cfg.finetuning.max_epochs,
        num_sanity_val_steps=0,
        logger=tb_logger,
        accumulate_grad_batches=cfg.Trainer.accumulate_grad_batches,
        callbacks=[ckpt_cb, lr_monitor],
        log_every_n_steps=1,
    )

    trainer.fit(rl_module, datamodule)


if __name__ == "__main__":
    main()
