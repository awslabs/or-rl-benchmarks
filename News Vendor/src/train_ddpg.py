from ray.tune.registry import register_env
from ray import tune

from sagemaker_rl.ray_launcher import SageMakerRayLauncher


class MyLauncher(SageMakerRayLauncher):

    def register_env_creator(self):
        from news_vendor_environment import NewsVendorGymEnvironment
        register_env("NewsVendorGymEnvironment-v1", lambda env_config: NewsVendorGymEnvironment(env_config))

    def get_experiment_config(self):
        max_iterations = 1000
        return {
            "training": {
                "env": "NewsVendorGymEnvironment-v1",
                "run": "DDPG",
                "config": self.__build_config(),
                "checkpoint_at_end": True,
                "stop": {
                    "training_iteration": max_iterations
                }
            }
        }

    @staticmethod
    def __build_config():
        env_config = {
            "lead_time": 2,
        }
        model_config = {
            # === Built-in options ===
            # Filter config. List of [out_channels, kernel, stride] for each filter
            "conv_filters": None,
            # Nonlinearity for built-in convnet
            "conv_activation": "relu",
            # Nonlinearity for fully connected net (tanh, relu)
            "fcnet_activation": "tanh",
            # Number of hidden layers for fully connected net
            "fcnet_hiddens": tune.grid_search([[256, 128, 64], [128, 64], [64, 32]]),
            # For control envs, documented in ray.rllib.models.Model
            "free_log_std": False,
            # (deprecated) Whether to use sigmoid to squash actions to space range
            "squash_to_range": False,
            # == LSTM ==
            # Whether to wrap the model with a LSTM
            "use_lstm": False,
            # Max seq len for training the LSTM, defaults to 20
            "max_seq_len": 20,
            # Size of the LSTM cell
            "lstm_cell_size": 256,
            # Whether to feed a_{t-1}, r_{t-1} to LSTM
            "lstm_use_prev_action_reward": False,

            # == Atari ==
            # Whether to enable framestack for Atari envs
            "framestack": True,
            # Final resized frame dimension
            "dim": 84,
            # (deprecated) Converts ATARI frame to 1 Channel Grayscale image
            "grayscale": False,
            # (deprecated) Changes frame to range from [-1, 1] if true
            "zero_mean": True,

            # === Options for custom models ===
            # Name of a custom preprocessor to use
            "custom_preprocessor": None,
            # Name of a custom model to use
            "custom_model": None,
            # Extra options to pass to the custom classes
            "custom_options": {},
        }

        ddpg_config = {
            # === Twin Delayed DDPG (TD3) and Soft Actor-Critic (SAC) tricks ===
            # TD3: https://spinningup.openai.com/en/latest/algorithms/td3.html
            # twin Q-net
            "twin_q": False,
            # delayed policy update
            "policy_delay": 1,
            # target policy smoothing
            # this also forces the use of gaussian instead of OU noise for exploration
            "smooth_target_policy": False,
            # gaussian stddev of act noise
            "act_noise": 0.1,
            # gaussian stddev of target noise
            "target_noise": 0.2,
            # target noise limit (bound)
            "noise_clip": 0.5,

            # === Model ===
            # Hidden layer sizes of the policy network
            "actor_hiddens": tune.grid_search([[64, 64], [32, 32]]),
            # Hidden layers activation of the policy network
            "actor_hidden_activation": "relu",
            # Hidden layer sizes of the critic network
            "critic_hiddens": tune.grid_search([[64, 64], [32, 32]]),
            # Hidden layers activation of the critic network
            "critic_hidden_activation": "relu",
            # N-step Q learning
            "n_step": 1,

            # === Exploration ===
            # Max num timesteps for annealing schedules. Exploration is annealed from
            # 1.0 to exploration_fraction over this number of timesteps scaled by
            # exploration_fraction
            "schedule_max_timesteps": 100000,
            # Number of env steps to optimize for before returning
            "timesteps_per_iteration": 1000,
            # Fraction of entire training period over which the exploration rate is
            # annealed
            "exploration_fraction": 0.1,
            # Final value of random action probability
            "exploration_final_eps": 0.02,
            # OU-noise scale
            "noise_scale": 0.1,
            # theta
            "exploration_theta": 0.15,
            # sigma
            "exploration_sigma": 0.2,
            # Update the target network every `target_network_update_freq` steps.
            "target_network_update_freq": 0,
            # Update the target by \tau * policy + (1-\tau) * target_policy
            "tau": 0.002,

            # === Replay buffer ===
            # Size of the replay buffer. Note that if async_updates is set, then
            # each worker will have a replay buffer of this size.
            "buffer_size": 50000,
            # If True prioritized replay buffer will be used.
            "prioritized_replay": True,
            # Alpha parameter for prioritized replay buffer.
            "prioritized_replay_alpha": 0.6,
            # Beta parameter for sampling from prioritized replay buffer.
            "prioritized_replay_beta": 0.4,
            # Epsilon to add to the TD errors when updating priorities.
            "prioritized_replay_eps": 1e-6,
            # Whether to LZ4 compress observations
            "compress_observations": False,

            # === Optimization ===
            # Learning rate for adam optimizer.
            # Instead of using two optimizers, we use two different loss coefficients
            "lr": 1e-3,
            "actor_loss_coeff": 0.1,
            "critic_loss_coeff": 1.0,
            # If True, use huber loss instead of squared loss for critic network
            # Conventionally, no need to clip gradients if using a huber loss
            "use_huber": False,
            # Threshold of a huber loss
            "huber_threshold": 1.0,
            # Weights for L2 regularization
            "l2_reg": 1e-6,
            # If not None, clip gradients during optimization at this value
            "grad_norm_clipping": None,
            # How many steps of the model to sample before learning starts.
            "learning_starts": 1500,
            # Update the replay buffer with this many samples at once. Note that this
            # setting applies per-worker if num_workers > 1.
            "sample_batch_size": 1,
            # Size of a batched sampled from replay buffer for training. Note that
            # if async_updates is set, then each worker returns gradients for a
            # batch of this size.
            "train_batch_size": 256,

            # === Parallelism ===
            # Number of workers for collecting samples with. This only makes sense
            # to increase if your environment is particularly slow to sample, or if
            # you"re using the Async or Ape-X optimizers.
            "num_workers": 0,
            # Optimizer class to use.
            "optimizer_class": "SyncReplayOptimizer",
            # Whether to use a distribution of epsilons across workers for exploration.
            "per_worker_exploration": False,
            # Whether to compute priorities on workers.
            "worker_side_prioritization": False,
            # Prevent iterations from going lower than this time span
            "min_iter_time_s": 1,
        }

        res_config = {
            # === Resources ===
            # Number of actors used for parallelism
            "num_workers": 1,
            # Number of GPUs to allocate to the driver. Note that not all algorithms
            # can take advantage of driver GPUs. This can be fraction (e.g., 0.3 GPUs).
            "num_gpus": 0,
            # Number of CPUs to allocate per worker.
            "num_cpus_per_worker": 2,
            # Number of GPUs to allocate per worker. This can be fractional.
            "num_gpus_per_worker": 0,
            # Any custom resources to allocate per worker.
            "custom_resources_per_worker": {},
            # Number of CPUs to allocate for the driver. Note: this only takes effect
            # when running in Tune.
            "num_cpus_for_driver": 2,
        }

        config = {}
        config.update(res_config)
        config.update(ddpg_config)
        config["env_config"] = env_config
        config["model"] = model_config

        return config


if __name__ == "__main__":
    MyLauncher().train_main()
