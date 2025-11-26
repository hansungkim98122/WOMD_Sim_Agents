import tensorflow as tf
from waymo_open_dataset.metrics.ops import py_metrics_ops
from waymo_open_dataset.protos import sim_agents_submission_pb2
from waymo_open_dataset.utils.sim_agents import submission_specs
from waymo_open_dataset.wdl_limited.sim_agents_metrics import metric_features
from waymo_open_dataset.wdl_limited.sim_agents_metrics import metrics

from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.protos.scenario_pb2 import Scenario

class MotionMetrics(tf.keras.metrics.Metric):
  """Wrapper for motion metrics computation."""

  def __init__(self, config):
    super().__init__()
    self._prediction_trajectory = []
    self._prediction_score = []
    self._ground_truth_trajectory = []
    self._ground_truth_is_valid = []
    self._prediction_ground_truth_indices = []
    self._prediction_ground_truth_indices_mask = []
    self._object_type = []
    self._metrics_config = config

  def reset_state(self):
    self._prediction_trajectory = []
    self._prediction_score = []
    self._ground_truth_trajectory = []
    self._ground_truth_is_valid = []
    self._prediction_ground_truth_indices = []
    self._prediction_ground_truth_indices_mask = []
    self._object_type = []

  def update_state(self, prediction_trajectory, prediction_score,
                   ground_truth_trajectory, ground_truth_is_valid,
                   prediction_ground_truth_indices,
                   prediction_ground_truth_indices_mask, object_type):
    self._prediction_trajectory.append(prediction_trajectory)
    self._prediction_score.append(prediction_score)
    self._ground_truth_trajectory.append(ground_truth_trajectory)
    self._ground_truth_is_valid.append(ground_truth_is_valid)
    self._prediction_ground_truth_indices.append(
        prediction_ground_truth_indices)
    self._prediction_ground_truth_indices_mask.append(
        prediction_ground_truth_indices_mask)
    self._object_type.append(object_type)

  def result(self):
    # [batch_size, num_preds, 1, 1, steps, 2].
    # The ones indicate top_k = 1, num_agents_per_joint_prediction = 1.
    prediction_trajectory = tf.concat(self._prediction_trajectory, 0)
    # [batch_size, num_preds, 1].
    prediction_score = tf.concat(self._prediction_score, 0)
    # [batch_size, num_agents, gt_steps, 7].
    ground_truth_trajectory = tf.concat(self._ground_truth_trajectory, 0)
    # [batch_size, num_agents, gt_steps].
    ground_truth_is_valid = tf.concat(self._ground_truth_is_valid, 0)
    # [batch_size, num_preds, 1].
    prediction_ground_truth_indices = tf.concat(
        self._prediction_ground_truth_indices, 0)
    # [batch_size, num_preds, 1].
    prediction_ground_truth_indices_mask = tf.concat(
        self._prediction_ground_truth_indices_mask, 0)
    # [batch_size, num_agents].
    object_type = tf.cast(tf.concat(self._object_type, 0), tf.int64)

    # We are predicting more steps than needed by the eval code. Subsample.
    interval = (
        self._metrics_config.track_steps_per_second //
        self._metrics_config.prediction_steps_per_second)
    prediction_trajectory = prediction_trajectory[...,
                                                  (interval - 1)::interval, :]

    return py_metrics_ops.motion_metrics(
        config=self._metrics_config.SerializeToString(),
        prediction_trajectory=prediction_trajectory,
        prediction_score=prediction_score,
        ground_truth_trajectory=ground_truth_trajectory,
        ground_truth_is_valid=ground_truth_is_valid,
        prediction_ground_truth_indices=prediction_ground_truth_indices,
        prediction_ground_truth_indices_mask=prediction_ground_truth_indices_mask,
        object_type=object_type)
  

#Sim Agents
def scenario_rollouts_from_states(
    scenario: scenario_pb2.Scenario, states: tf.Tensor, object_ids: tf.Tensor
) -> sim_agents_submission_pb2.ScenarioRollouts:
  # States shape: (num_rollouts, num_objects, num_steps, 4).
  # Objects IDs shape: (num_objects,).
  joint_scenes = []
  for i_rollout in range(states.shape[0]):
    joint_scenes.append(joint_scene_from_states(states[i_rollout], object_ids))
  return sim_agents_submission_pb2.ScenarioRollouts(
      # Note: remember to include the Scenario ID in the proto message.
      joint_scenes=joint_scenes,
      scenario_id=scenario.scenario_id,
  )

def joint_scene_from_states(
    states: tf.Tensor, object_ids: tf.Tensor
) -> sim_agents_submission_pb2.JointScene:
  # States shape: (num_objects, num_steps, 4).
  # Objects IDs shape: (num_objects,).
  states = states.numpy()
  simulated_trajectories = []
  for i_object in range(len(object_ids)):
    simulated_trajectories.append(
        sim_agents_submission_pb2.SimulatedTrajectory(
            center_x=states[i_object, :, 0],
            center_y=states[i_object, :, 1],
            center_z=states[i_object, :, 2],
            heading=states[i_object, :, 3],
            object_id=object_ids[i_object],
        )
    )
  return sim_agents_submission_pb2.JointScene(
      simulated_trajectories=simulated_trajectories
  )

def scenario_rollout_generation(scenario, challenge_type, simulated_states, logged_trajectories):
    # Package the first simulation into a `JointScene`
    joint_scene = joint_scene_from_states(
        simulated_states[0, :, :, :], logged_trajectories.object_id
    )
    # Validate the joint scene. Should raise an exception if it's invalid.
    submission_specs.validate_joint_scene(joint_scene, scenario, challenge_type)

    scenario_rollouts = scenario_rollouts_from_states(
        scenario, simulated_states, logged_trajectories.object_id
    )
    # As before, we can validate the message we just generate.
    submission_specs.validate_scenario_rollouts(scenario_rollouts, scenario)

    return scenario_rollouts, joint_scene

def sim_agent_evaluation(scenario, challenge_type, simulated_states, logged_trajectories):
    scenario_rollouts, joint_scene = scenario_rollout_generation(scenario, challenge_type, simulated_states, logged_trajectories)

    # Compute the features for a single JointScene.
    single_scene_features = metric_features.compute_metric_features(
        scenario, joint_scene
    )

    # These features will be computed only for the `tracks_to_predict` objects.
    print(
        'Evaluated objects:'
        f' {submission_specs.get_evaluation_sim_agent_ids(scenario, challenge_type)}'
    )
    # This will also match single_scene_features.object_ids
    print(f'Evaluated objects in features: {single_scene_features.object_id}')

    # Features contain a validity flag, which for simulated rollouts must be always
    # True, because we are requiring the sim agents to be always valid when replaced.
    print(f'Are all agents valid: {tf.reduce_all(single_scene_features.valid)}')

    # ============ FEATURES ============
    # Average displacement feature. This corresponds to ADE in the BP challenges,
    # here is used just as a comparison (it's not actually included in the final score).
    # Shape: (1, n_objects).
    print(
        f'ADE: {tf.reduce_mean(single_scene_features.average_displacement_error)}'
    )

    # Load the test configuration.
    config = metrics.load_metrics_config(challenge_type)

    scenario_metrics = metrics.compute_scenario_metrics_for_bundle(
        config, scenario, scenario_rollouts
    )
    print(scenario_metrics)