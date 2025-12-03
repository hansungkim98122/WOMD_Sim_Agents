from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger  

from utils.config import load_config_act
from datamodule.datamodule import MultiDataModule
from model.smart import SMART
from utils.logging import Logging


if __name__ == '__main__':
    parser = ArgumentParser()
    Predictor_hash = {"smart": SMART, }
    parser.add_argument('--config', type=str, default='configs/training/training.yaml')
    parser.add_argument('--pretrain_ckpt', type=str, default="")
    # NOTE: this ckpt_path is for *resuming* training, not for saving checkpoints
    parser.add_argument('--ckpt_path', type=str, default="")
    parser.add_argument('--save_ckpt_path', type=str, default="checkpoints")  # default dir for ckpts
    parser.add_argument('--log_dir', type=str, default="tb_logs")  # <-- add this if you want
    args = parser.parse_args()

    config = load_config_act(args.config)

    Predictor = Predictor_hash[config.Model.predictor]
    strategy = DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True)

    Data_config = config.Dataset
    Data_config['smart_token'] = config.Model.use_smart_tokens
    datamodule = MultiDataModule(**vars(Data_config))

    if args.pretrain_ckpt == "":
        model = Predictor(config.Model)
    else:
        logger_std = Logging().log(level='DEBUG')
        model = Predictor(config.Model)
        model.load_params_from_file(filename=args.pretrain_ckpt,
                                    logger=logger_std)

    # ----------------- Logger (TensorBoard) -----------------
    tb_logger = TensorBoardLogger(
        save_dir=args.log_dir,  # root dir
        name="smart",           # experiment name
        # version can be int/str or None -> auto-increment
    )

    # ----------------- Checkpoints -----------------
    # If you don't pass dirpath, it will default to <logger.log_dir>/checkpoints
    model_checkpoint = ModelCheckpoint(
        dirpath=args.save_ckpt_path,    # folder where ckpts go
        filename="{epoch:02d}-{val_cls_acc:.4f}",  # pattern
        monitor='val_cls_acc',          # metric to watch
        every_n_epochs=1,
        save_top_k=5,
        mode='max',
        save_last=True,                 # also keep last.ckpt
    )

    # LR logging
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer_config = config.Trainer
    trainer = pl.Trainer(
        accelerator=trainer_config.accelerator,
        devices=trainer_config.devices,
        strategy=strategy,
        accumulate_grad_batches=trainer_config.accumulate_grad_batches,
        num_nodes=trainer_config.num_nodes,
        callbacks=[model_checkpoint, lr_monitor],
        max_epochs=trainer_config.max_epochs,
        num_sanity_val_steps=0,
        gradient_clip_val=0.5,
        logger=tb_logger,  # <-- hook TensorBoard into Trainer
    )

    # ----------------- Train / Resume -----------------
    if args.ckpt_path == "":
        trainer.fit(model, datamodule)
    else:
        # here ckpt_path is a PL checkpoint path (e.g., from ModelCheckpoint)
        trainer.fit(model, datamodule, ckpt_path=args.ckpt_path)
