import tensorflow as tf
from waymo_open_dataset.protos import motion_metrics_pb2
from google.protobuf import text_format
import torch as th
from datetime import datetime
import os
from tqdm import tqdm
from eval_utils import scenario_rollout_generation
from waymo_open_dataset.wdl_limited.sim_agents_metrics import metric_features
from waymo_open_dataset.wdl_limited.sim_agents_metrics import metrics
from waymo_open_dataset.utils.sim_agents import submission_specs

def womd_scenario2tfexample():
    num_map_samples = 30000
    # Example field definition
    roadgraph_features = {
        'roadgraph_samples/dir': tf.io.FixedLenFeature(
            [num_map_samples, 3], tf.float32, default_value=None
        ),
        'roadgraph_samples/id': tf.io.FixedLenFeature(
            [num_map_samples, 1], tf.int64, default_value=None
        ),
        'roadgraph_samples/type': tf.io.FixedLenFeature(
            [num_map_samples, 1], tf.int64, default_value=None
        ),
        'roadgraph_samples/valid': tf.io.FixedLenFeature(
            [num_map_samples, 1], tf.int64, default_value=None
        ),
        'roadgraph_samples/xyz': tf.io.FixedLenFeature(
            [num_map_samples, 3], tf.float32, default_value=None
        ),
    }
    # Features of other agents.
    state_features = {
        'state/id':
            tf.io.FixedLenFeature([128], tf.float32, default_value=None),
        'state/type':
            tf.io.FixedLenFeature([128], tf.float32, default_value=None),
        'state/is_sdc':
            tf.io.FixedLenFeature([128], tf.int64, default_value=None),
        'state/tracks_to_predict':
            tf.io.FixedLenFeature([128], tf.int64, default_value=None),
        'state/current/bbox_yaw':
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        'state/current/height':
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        'state/current/length':
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        'state/current/timestamp_micros':
            tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
        'state/current/valid':
            tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
        'state/current/vel_yaw':
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        'state/current/velocity_x':
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        'state/current/velocity_y':
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        'state/current/width':
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        'state/current/x':
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        'state/current/y':
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        'state/current/z':
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        'state/future/bbox_yaw':
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        'state/future/height':
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        'state/future/length':
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        'state/future/timestamp_micros':
            tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
        'state/future/valid':
            tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
        'state/future/vel_yaw':
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        'state/future/velocity_x':
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        'state/future/velocity_y':
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        'state/future/width':
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        'state/future/x':
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        'state/future/y':
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        'state/future/z':
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        'state/past/bbox_yaw':
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        'state/past/height':
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        'state/past/length':
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        'state/past/timestamp_micros':
            tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
        'state/past/valid':
            tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
        'state/past/vel_yaw':
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        'state/past/velocity_x':
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        'state/past/velocity_y':
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        'state/past/width':
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        'state/past/x':
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        'state/past/y':
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        'state/past/z':
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    }

    traffic_light_features = {
        'traffic_light_state/current/state':
            tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
        'traffic_light_state/current/valid':
            tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
        'traffic_light_state/current/x':
            tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
        'traffic_light_state/current/y':
            tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
        'traffic_light_state/current/z':
            tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
        'traffic_light_state/past/state':
            tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
        'traffic_light_state/past/valid':
            tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
        'traffic_light_state/past/x':
            tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
        'traffic_light_state/past/y':
            tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
        'traffic_light_state/past/z':
            tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
    }

    features_description = {}
    features_description.update(roadgraph_features)
    features_description.update(state_features)
    features_description.update(traffic_light_features)

    return features_description

def parse(value):
  features_description = womd_scenario2tfexample()
  decoded_example = tf.io.parse_single_example(value, features_description)

  past_states = tf.stack([
      decoded_example['state/past/x'], decoded_example['state/past/y'],
      decoded_example['state/past/length'], decoded_example['state/past/width'],
      decoded_example['state/past/bbox_yaw'],
      decoded_example['state/past/velocity_x'],
      decoded_example['state/past/velocity_y']
  ], -1)

  cur_states = tf.stack([
      decoded_example['state/current/x'], decoded_example['state/current/y'],
      decoded_example['state/current/length'],
      decoded_example['state/current/width'],
      decoded_example['state/current/bbox_yaw'],
      decoded_example['state/current/velocity_x'],
      decoded_example['state/current/velocity_y']
  ], -1)

  input_states = tf.concat([past_states, cur_states], 1)[..., :2]

  future_states = tf.stack([
      decoded_example['state/future/x'], decoded_example['state/future/y'],
      decoded_example['state/future/length'],
      decoded_example['state/future/width'],
      decoded_example['state/future/bbox_yaw'],
      decoded_example['state/future/velocity_x'],
      decoded_example['state/future/velocity_y']
  ], -1)

  gt_future_states = tf.concat([past_states, cur_states, future_states], 1)

  past_is_valid = decoded_example['state/past/valid'] > 0
  current_is_valid = decoded_example['state/current/valid'] > 0
  future_is_valid = decoded_example['state/future/valid'] > 0
  gt_future_is_valid = tf.concat(
      [past_is_valid, current_is_valid, future_is_valid], 1)

  # If a sample was not seen at all in the past, we declare the sample as
  # invalid.
  sample_is_valid = tf.reduce_any(
      tf.concat([past_is_valid, current_is_valid], 1), 1)

  inputs = {
      'input_states': input_states,
      'gt_future_states': gt_future_states,
      'gt_future_is_valid': gt_future_is_valid,
      'object_type': decoded_example['state/type'],
      'tracks_to_predict': decoded_example['state/tracks_to_predict'] > 0,
      'sample_is_valid': sample_is_valid,
  }
  return inputs

def default_metrics_config():
  config = motion_metrics_pb2.MotionMetricsConfig()
  config_text = """
  track_steps_per_second: 10
  prediction_steps_per_second: 2
  track_history_samples: 10
  track_future_samples: 80
  speed_lower_bound: 1.4
  speed_upper_bound: 11.0
  speed_scale_lower: 0.5
  speed_scale_upper: 1.0
  step_configurations {
    measurement_step: 5
    lateral_miss_threshold: 1.0
    longitudinal_miss_threshold: 2.0
  }
  step_configurations {
    measurement_step: 9
    lateral_miss_threshold: 1.8
    longitudinal_miss_threshold: 3.6
  }
  step_configurations {
    measurement_step: 15
    lateral_miss_threshold: 3.0
    longitudinal_miss_threshold: 6.0
  }
  max_predictions: 6
  """
  text_format.Parse(config_text, config)
  return config


def tf2torch(t: tf.Tensor) -> th.Tensor:
    """
    Zero-copy when possible. Preserves device and dtype.
    NOTE: no autograd across frameworks.
    """
    dlpack = tf.experimental.dlpack.to_dlpack(t)     # TF -> DLPack capsule
    return th.utils.dlpack.from_dlpack(dlpack)    # DLPack -> torch


def torch2tf(t: th.Tensor) -> tf.Tensor:
        """
        Convert a PyTorch tensor to a TensorFlow tensor via DLPack.
        Detach and make contiguous where appropriate before conversion.
        """
        # Detach (no grad) should be done by caller; ensure contiguous for DLPack.
        if t.requires_grad:
                t = t.detach()
        t = t.contiguous()
        dlpack = th.utils.dlpack.to_dlpack(t)
        return tf.experimental.dlpack.from_dlpack(dlpack)

def dict2torchdict(x: dict):
   for key in x.keys():
       x[key] = tf2torch(x[key])
   return x

def get_model_param_str(model_cfg):
    if model_cfg.name == 'MLP':
        return 'MLP_' + 'L' + str(model_cfg.num_layers) + '_H' + str(model_cfg.hidden_dim) + '_' + model_cfg.activation + '_D' + str(model_cfg.dropout)
    else:
        return 'Backbone' + 'L' + str(model_cfg.num_layers) + '_H' + str(model_cfg.hidden_dim) + '_' + model_cfg.activation + '_D' + str(model_cfg.dropout)

def get_savedirs(cfg):
    current_datetime = datetime.now()
    model_save_dir = cfg.model_save_dir + '/' + get_model_param_str(cfg.model) + '_' +current_datetime.strftime("%Y-%m-%d-%H:%M:%S") + '/' 
    if not os.path.isdir(model_save_dir):
        os.mkdir(model_save_dir)
    model_name = get_model_param_str(cfg.model) + '_' +current_datetime.strftime("%Y-%m-%d-%H:%M:%S")
    tensorboard_log_dir = cfg.tensorboard_save_dir + '/' + get_model_param_str(cfg.model) + '_' + current_datetime.strftime("%Y-%m-%d-%H:%M:%S")
    config_save_path = model_save_dir + 'config.yaml'
    return tensorboard_log_dir, model_save_dir, model_name, config_save_path

def train_step(inputs, model, loss_fn, optimizer, motion_metrics, metrics_config, *, device=None):
    """
    Args
    - inputs: dict with keys
        'input_states'               : [B, num_agents, D]           (float)
        'gt_future_states'           : [B, num_agents, gt_steps, 7] (float)
        'gt_future_is_valid'         : [B, num_agents, gt_steps]    (bool/int/float)
        'tracks_to_predict'          : [B, num_agents]              (bool/int)
        'object_type'                : [B, num_agents]              (int64)
    - model: th.nn.Module, returns pred_trajectory [B, num_agents, steps, 2]
    - loss_fn: th loss with reduction='none' (e.g., nn.SmoothL1Loss(reduction='none') or nn.MSELoss(reduction='none'))
    - optimizer: th optimizer
    - motion_metrics: your MotionMetricsth wrapper
    - metrics_config: has attribute `track_history_samples`
    - device: optional th device

    Returns
    - loss_value (float)
    """
    if device is None:
        device = next(model.parameters()).device

    model.train()
    optimizer.zero_grad(set_to_none=True)

    # --- Move inputs to device ---
    inputs = dict2torchdict(inputs)
    states              = inputs['input_states'].to(device)                 # [B, A, D]
    gt_trajectory       = inputs['gt_future_states'].to(device)             # [B, A, gt_steps, 7]
    gt_is_valid         = inputs['gt_future_is_valid'].to(device)           # [B, A, gt_steps]
    tracks_to_predict   = inputs['tracks_to_predict'].to(device)            # [B, A]
    object_type         = inputs['object_type'].to(device)                  # [B, A] (int)

    # --- Forward ---
    # pred_trajectory: [B, A, steps, 2]
    pred_trajectory = model(states)

    # --- Targets and weights (match TF logic) ---
    prediction_start = int(metrics_config.track_history_samples) + 1

    # gt targets: [B, A, steps, 2]
    gt_targets = gt_trajectory[..., prediction_start:, :2]

    # weights: [B, A, steps] = valid * tracks_to_predict[:, :, None]
    weights = gt_is_valid[..., prediction_start:].to(th.float32) * tracks_to_predict.unsqueeze(-1).to(th.float32)

    # --- Weighted loss ---
    # loss_fn should output per-element loss with reduction='none'
    # shape usually [B, A, steps, 2]
    per_elem = loss_fn(pred_trajectory, gt_targets)

    # broadcast weights over the last coord dim (x,y)
    w = weights.unsqueeze(-1)  # [B, A, steps, 1]
    weighted = per_elem * w

    denom = w.sum().clamp_min(1.0)
    loss = weighted.sum() / denom

    # --- Backprop / step ---
    loss.backward()
    optimizer.step()

    # --- Prepare tensors for motion metrics (no grad) ---
    with th.no_grad():
        # [B, A, steps, 2] -> [B, A, 1, 1, steps, 2]
        pred_for_metrics = pred_trajectory.unsqueeze(2).unsqueeze(2)  # add top_k=1 and num_agents_per_joint_prediction=1

        # Fake score: ones [B, A, 1]
        pred_score = th.ones(pred_for_metrics.shape[:3], device=device)

        B = tracks_to_predict.shape[0]
        A = tracks_to_predict.shape[1]

        # pred_gt_indices: [B, A, 1], where each agent aligns with its own GT index
        # same as TF: tile(range(num_agents), [batch, 1, 1])
        pred_gt_indices = th.arange(A, dtype=th.long, device=device).view(1, A, 1).expand(B, A, 1)

        # mask: [B, A, 1]
        pred_gt_indices_mask = tracks_to_predict.unsqueeze(-1)

        # Convert tensors to TensorFlow (detach first) and update metrics.
        # Cast to the expected dtypes required by the MotionMetrics op
        # (e.g., bool for validity masks, int64 for indices/object_type).
        pred_traj_tf = torch2tf(pred_for_metrics)
        pred_score_tf = torch2tf(pred_score)
        gt_traj_tf = torch2tf(gt_trajectory)
        gt_is_valid_tf = tf.cast(torch2tf(gt_is_valid), tf.bool)
        pred_gt_indices_tf = tf.cast(torch2tf(pred_gt_indices), tf.int64)
        pred_gt_indices_mask_tf = tf.cast(torch2tf(pred_gt_indices_mask), tf.bool)
        object_type_tf = tf.cast(torch2tf(object_type), tf.int64)

        motion_metrics.update_state(
            prediction_trajectory=pred_traj_tf,
            prediction_score=pred_score_tf,
            ground_truth_trajectory=gt_traj_tf,
            ground_truth_is_valid=gt_is_valid_tf,
            prediction_ground_truth_indices=pred_gt_indices_tf,
            prediction_ground_truth_indices_mask=pred_gt_indices_mask_tf,
            object_type=object_type_tf,
        )

    return float(loss.item())

def validation_step(inputs, model, loss_fn, motion_metrics, metrics_config, *, device=None):
    """
    Args
    - inputs: dict with keys
        'input_states'               : [B, num_agents, D]           (float)
        'gt_future_states'           : [B, num_agents, gt_steps, 7] (float)
        'gt_future_is_valid'         : [B, num_agents, gt_steps]    (bool/int/float)
        'tracks_to_predict'          : [B, num_agents]              (bool/int)
        'object_type'                : [B, num_agents]              (int64)
    - model: th.nn.Module, returns pred_trajectory [B, num_agents, steps, 2]
    - loss_fn: th loss with reduction='none' (e.g., nn.SmoothL1Loss(reduction='none') or nn.MSELoss(reduction='none'))
    - motion_metrics: your MotionMetricsth wrapper
    - metrics_config: has attribute `track_history_samples`
    - device: optional th device

    Returns
    - loss_value (float)
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    
    # --- Move inputs to device ---
    inputs = dict2torchdict(inputs)
    states              = inputs['input_states'].to(device)                 # [B, A, D]
    gt_trajectory       = inputs['gt_future_states'].to(device)             # [B, A, gt_steps, 7]
    gt_is_valid         = inputs['gt_future_is_valid'].to(device)           # [B, A, gt_steps]
    tracks_to_predict   = inputs['tracks_to_predict'].to(device)            # [B, A]
    object_type         = inputs['object_type'].to(device)                  # [B, A] (int)

    # --- Forward ---
    # pred_trajectory: [B, A, steps, 2]
    with th.no_grad():
        pred_trajectory = model(states)

        # --- Targets and weights (match TF logic) ---
        prediction_start = int(metrics_config.track_history_samples) + 1

        # gt targets: [B, A, steps, 2]
        gt_targets = gt_trajectory[..., prediction_start:, :2]

        # weights: [B, A, steps] = valid * tracks_to_predict[:, :, None]
        weights = gt_is_valid[..., prediction_start:].to(th.float32) * tracks_to_predict.unsqueeze(-1).to(th.float32)

        # --- Weighted loss ---
        # loss_fn should output per-element loss with reduction='none'
        # shape usually [B, A, steps, 2]
        per_elem = loss_fn(pred_trajectory, gt_targets)

        # broadcast weights over the last coord dim (x,y)
        w = weights.unsqueeze(-1)  # [B, A, steps, 1]
        weighted = per_elem * w

        denom = w.sum().clamp_min(1.0)
        loss = weighted.sum() / denom

        # [B, A, steps, 2] -> [B, A, 1, 1, steps, 2]
        pred_for_metrics = pred_trajectory.unsqueeze(2).unsqueeze(2)  # add top_k=1 and num_agents_per_joint_prediction=1

        # Fake score: ones [B, A, 1]
        pred_score = th.ones(pred_for_metrics.shape[:3], device=device)

        B = tracks_to_predict.shape[0]
        A = tracks_to_predict.shape[1]

        # pred_gt_indices: [B, A, 1], where each agent aligns with its own GT index
        # same as TF: tile(range(num_agents), [batch, 1, 1])
        pred_gt_indices = th.arange(A, dtype=th.long, device=device).view(1, A, 1).expand(B, A, 1)

        # mask: [B, A, 1]
        pred_gt_indices_mask = tracks_to_predict.unsqueeze(-1)

        # Convert tensors to TensorFlow (detach first) and update metrics.
        # Cast to the expected dtypes required by the MotionMetrics op
        # (e.g., bool for validity masks, int64 for indices/object_type).
        pred_traj_tf = torch2tf(pred_for_metrics)
        pred_score_tf = torch2tf(pred_score)
        gt_traj_tf = torch2tf(gt_trajectory)
        gt_is_valid_tf = tf.cast(torch2tf(gt_is_valid), tf.bool)
        pred_gt_indices_tf = tf.cast(torch2tf(pred_gt_indices), tf.int64)
        pred_gt_indices_mask_tf = tf.cast(torch2tf(pred_gt_indices_mask), tf.bool)
        object_type_tf = tf.cast(torch2tf(object_type), tf.int64)

        motion_metrics.update_state(
            prediction_trajectory=pred_traj_tf,
            prediction_score=pred_score_tf,
            ground_truth_trajectory=gt_traj_tf,
            ground_truth_is_valid=gt_is_valid_tf,
            prediction_ground_truth_indices=pred_gt_indices_tf,
            prediction_ground_truth_indices_mask=pred_gt_indices_mask_tf,
            object_type=object_type_tf,
        )
    model.train()
    return float(loss.item())

#Motion Prediction
def cross_validation(epoch, validation_dataset, model, loss_fn, motion_metrics, metrics_config, metric_names, writer):
    print('Starting cross-validation...') 
    for _, val_batch in tqdm(enumerate(validation_dataset)):
        with th.no_grad():
            val_loss = validation_step(val_batch, model, loss_fn, motion_metrics, metrics_config)

    validation_metric_values = motion_metrics.result().numpy()
    # Log to tensorboard
    print('Validation Metrics:')
    for i, m in enumerate(['min_ade', 'min_fde', 'miss_rate', 'overlap_rate', 'map']):
        for j, n in enumerate(metric_names):
            val = float(validation_metric_values[i, j])
            print('\t{}/{}: {}'.format(m, n, val))
            writer.add_scalar('motion_metrics/validation/'+ m +'/'+n, val)
    writer.add_scalar("", epoch, val_loss)
    return float(val_loss.item())

#Sim Agents
