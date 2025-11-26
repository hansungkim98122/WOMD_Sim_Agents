import hydra
from hydra.utils import instantiate
import torch as th
from omegaconf import OmegaConf
import tensorflow_datasets as tfds
from torch.utils.tensorboard import SummaryWriter
import logging
from waymo_open_dataset.metrics.python import config_util_py as config_util
from utils.train_utils import get_savedirs, default_metrics_config, parse, train_step, cross_validation
from utils.eval_utils import MotionMetrics
from utils.dataset_utils import load_dataset, visualize_scenario_sim_agents
from tqdm import tqdm
from waymo_open_dataset.utils.sim_agents import submission_specs
from tokens.road_vector_tokenizer import RoadVectorTokenizer   

log = logging.getLogger(__name__)
tfds.display_progress_bar(enable=True)

@hydra.main(version_base="1.2", config_path="training_configs", config_name="mlp.yaml")
def train(cfg):
    
    tensorboard_log_dir, model_save_dir, model_name, config_save_path = get_savedirs(cfg)
    log.info(OmegaConf.to_yaml(cfg))
    OmegaConf.save(cfg, config_save_path)
    writer = SummaryWriter(log_dir = tensorboard_log_dir)

    # Load dataset using default features mapping (TensorFlow)
    training_dataset, validation_dataset, test_dataset, features_description = load_dataset(cfg.dataset.filepath)

    # Print the submission specs
    challenge_type = submission_specs.ChallengeType.SIM_AGENTS
    submission_config = submission_specs.get_submission_config(challenge_type)

    print(f'Simulation length, in steps: {submission_config.n_simulation_steps}')
    print(
        'Duration of a step, in seconds:'
        f' {submission_config.step_duration_seconds}s (frequency:'
        f' {1/submission_config.step_duration_seconds}Hz)'
    )
    print(
        'Number of parallel simulations per Scenario:'
        f' {submission_config.n_rollouts}'
    )
    visualize_scenario_sim_agents(validation_dataset)

    # Define the model and optimizer
    model = instantiate(cfg.model)
    optimizer = th.optim.AdamW(model.parameters(),lr=cfg.learning_rate)
    scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs,eta_min=cfg.learning_rate * 0.01)
    loss_fn = th.nn.MSELoss()

    metrics_config = default_metrics_config()
    
    motion_metrics = MotionMetrics(metrics_config)
    metric_names = config_util.get_breakdown_names_from_motion_config(
        metrics_config)
    
    #Setup dataset
    # training_dataset = training_dataset.batch(cfg.batch_size)

    num_steps = 0
    model_save_freq = cfg.model_save_freq
    cross_validation_freq = cfg.cross_validation_freq

    for epoch in tqdm(range(cfg.epochs), desc='Epoch: '):
        # Iterate over the batches of the dataset.
        for step, batch in tqdm(enumerate(training_dataset), desc=f'Batch Iteration [{cfg.batch_size}]: '):
            #Tensorflow to torch tensor
            loss = train_step(batch, model, loss_fn, optimizer, motion_metrics, metrics_config,device =th.device("cpu"))
            writer.add_scalar("loss/train", loss, epoch)
            num_steps += step * int(cfg.batch_size)

        # Display metrics at the end of each epoch.
        print('Computing Training Metrics...')
        train_metric_values = motion_metrics.result().numpy()

        # Log to tensorboard
        print('Training Metrics:')
        for i, m in enumerate(['min_ade', 'min_fde', 'miss_rate', 'overlap_rate', 'map']):
            for j, n in enumerate(metric_names):
                val = float(train_metric_values[i, j])
                print('\t{}/{}: {}'.format(m, n, val))
                writer.add_scalar('motion_metrics/train/'+ m +'/'+n, val)
        writer.add_scalar("num_steps", epoch, num_steps)

        # Cross-validation
        if epoch > 0 and epoch % cross_validation_freq == 0:
            cross_validation(epoch, 
                             validation_dataset, 
                             model,
                             loss_fn, 
                             motion_metrics, 
                             metrics_config, 
                             metric_names, 
                             writer)

        #Save the model
        if epoch % model_save_freq == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': float(loss),
            }
            th.save(checkpoint, model_save_dir + model_name + '_epoch_' +str(epoch) + '.pth')
        scheduler.step()
    writer.flush()
    print(''.center(80,'#'))

if __name__=='__main__':
    train()