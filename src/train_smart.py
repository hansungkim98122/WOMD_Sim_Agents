import os
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger

from datamodule.datamodule import MultiDataModule
from model.smart import SMART
# Assuming 'utils.logging' and 'Logging' are available as per your snippet
from utils.logging import Logging

# Map model names to classes
Predictor_hash = {
    "smart": SMART,
}

@hydra.main(version_base=None, config_path="configs/training", config_name="training")
def main(cfg: DictConfig):
    # 1. Save the configuration for reproducibility
    # Hydra automatically creates an output directory for each run.
    # We can save the resolved config there.
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    config_path = os.path.join(output_dir, "training_config.yaml")
    
    with open(config_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    print(f"Training configuration saved to: {config_path}")

    # 2. Setup Logging (Optional custom logger if needed, otherwise rely on PL)
    # Logging().log(level='DEBUG') 

    # 3. Initialize DataModule
    # Convert OmegaConf object to primitive dict for kwargs unpacking
    data_config = OmegaConf.to_container(cfg.Dataset, resolve=True)
    # Inject smart_token flag from Model config if needed
    data_config['smart_token'] = cfg.Model.use_smart_tokens
    
    datamodule = MultiDataModule(**data_config)

    # 4. Initialize Model
    predictor_name = cfg.Model.predictor
    if predictor_name not in Predictor_hash:
        raise ValueError(f"Unknown predictor: {predictor_name}. Available: {list(Predictor_hash.keys())}")
    
    PredictorClass = Predictor_hash[predictor_name]
    
    # Handle pretraining checkpoint loading
    pretrain_ckpt = cfg.get("pretrain_ckpt", "")
    if pretrain_ckpt:
        print(f"Loading pretrained model from: {pretrain_ckpt}")
        model = PredictorClass.load_from_checkpoint(pretrain_ckpt, config=cfg.Model) 
        # Note: Depending on your model implementation, you might need to pass config 
        # either to __init__ or ensure load_from_checkpoint handles it correctly.
        # If __init__ takes 'config', load_from_checkpoint usually re-uses hparams.
        # If it doesn't work, instantiate first and then load_state_dict.
    else:
        model = PredictorClass(cfg.Model)

    # 5. Setup Trainer Components
    strategy = DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True)
    
    # Logger
    # Use a specific log_dir from config or default to hydra output
    log_dir = cfg.get("log_dir", "tb_logs")
    tb_logger = TensorBoardLogger(
        save_dir=log_dir,
        name="smart",
        version=None # Let PL handle versioning or use hydra job id
    )

    # Checkpoints
    # Save checkpoints in the hydra output directory or a specified path
    # 'save_ckpt_path' should be defined in your yaml or passed as an override
    save_ckpt_path = cfg.get("save_ckpt_path", os.path.join(output_dir, "checkpoints"))
    
    model_checkpoint = ModelCheckpoint(
        dirpath=save_ckpt_path,
        filename="{epoch:02d}-{val_acc:.4f}",
        monitor='val_acc',
        every_n_epochs=1,
        save_top_k=5,
        mode='max',
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Trainer
    trainer_config = cfg.Trainer
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
        logger=tb_logger,
        # Hydra changes working dir, ensuring paths are correct relative to original cwd if needed
        # or use hydra.utils.to_absolute_path() for external files
    )

    # 6. Train
    # If resuming from a specific checkpoint (not just loading weights), pass it here
    # Assuming 'pretrain_ckpt' is for weights initialization, not resuming training state.
    # If resuming full state, use ckpt_path in fit().
    
    trainer.fit(model, datamodule)


if __name__ == '__main__':
    main()