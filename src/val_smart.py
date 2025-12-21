
from argparse import ArgumentParser
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from dataset.dataset import MultiDataset
from model.smart import SMART
from transforms.target_builder import WaymoTargetBuilder
from utils.config import load_config_act
from utils.logging import Logging

import tensorflow as tf
# Hide GPU from TensorFlow so it doesn't reserve VRAM.
tf.config.set_visible_devices([], 'GPU')

if __name__ == '__main__':
    pl.seed_everything(2, workers=True)
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/validation/validation.yaml")
    parser.add_argument('--pretrain_ckpt', type=str, default="")
    parser.add_argument('--ckpt_path', type=str, default="")
    parser.add_argument('--save_ckpt_path', type=str, default="")
    args = parser.parse_args()
    config = load_config_act(args.config)

    data_config = config.Dataset
    data_config['smart_token'] = config.Model.use_smart_tokens
    val_dataset = {
        "scalable": MultiDataset,
    }[data_config.dataset](root=data_config.root, split='val',
                           raw_dir=data_config.val_raw_dir,
                           processed_dir=data_config.val_processed_dir,
                           transform=WaymoTargetBuilder(config.Model.num_historical_steps, config.Model.decoder.num_future_steps),token_size=config.time_info.token_size)
    dataloader = DataLoader(val_dataset, batch_size=data_config.batch_size, shuffle=False, num_workers=data_config.num_workers,
                            pin_memory=data_config.pin_memory, persistent_workers=True if data_config.num_workers > 0 else False)
    Predictor = SMART
    if args.pretrain_ckpt == "":
        ValueError('Ckpt not defined')
    else:
        logger = Logging().log(level='DEBUG')
        model = Predictor(config.Model)
        print(f"Loading pretrained model from: {args.pretrain_ckpt}")
        model = Predictor.load_from_checkpoint(args.pretrain_ckpt, config=config.Model) 

    trainer_config = config.Trainer
    trainer = pl.Trainer(accelerator=trainer_config.accelerator,
                         devices=trainer_config.devices,
                         strategy='ddp', num_sanity_val_steps=0)
    trainer.validate(model, dataloader)