# configs/simple_tag_config.py
from ray.rllib.algorithms.maddpg import MADDPGConfig
from ray.rllib.policy.policy import PolicySpec

# --- Environment Configuration ---
ENV_CONFIG = {
    "grid_size": 7,
    "max_cycles": 100,
    # "render_mode": "human" # Set this in the training script if needed for debugging
}

# --- Algorithm Configuration ---
def get_maddpg_config(env_name_to_register, temp_env_for_spaces):
    """
    Generates the MADDPG configuration.
    Requires a temporary environment instance to get observation and action spaces.
    """
    # For MADDPG, policies are typically created for each agent.
    # RLlib's MADDPG will try to auto-populate policies based on env.possible_agents
    # if policies dict is not fully specified or if policy_mapping_fn returns agent_ids
    # that are not in policies. It's good practice to define them explicitly.

    policies = {
        agent_id: PolicySpec(
            observation_space=temp_env_for_spaces.observation_space(agent_id),
            action_space=temp_env_for_spaces.action_space(agent_id),
        )
        for agent_id in temp_env_for_spaces.possible_agents
    }

    # Policy mapping function. For MADDPG, each agent usually has its own policy.
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        return agent_id # agent_id will be "pursuer_0" or "evader_0"

    maddpg_config = (
        MADDPGConfig()
        .environment(env=env_name_to_register, env_config=ENV_CONFIG)
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
        .framework("torch") # or "tf2"
        .training(
            actor_lr=1e-4,
            critic_lr=1e-3,
            n_step=1, # Default is 3, 1 can sometimes be more stable for simpler envs
            tau=0.01,
            # Add other MADDPG specific parameters as needed
            # model = {"custom_model": "your_custom_model_if_any"}
        )
        .rollouts(
            num_rollout_workers=0, # Set to >0 for parallel sampling
            rollout_fragment_length="auto", # or a specific integer
        )
        .debugging(log_level="INFO") # "WARN" or "ERROR" for less verbose
        .resources(num_gpus=0) # Adjust if GPUs are available
    )
    return maddpg_config

# Example usage (will be used in train_maddpg.py)
# from environments import SimpleTagEnv # Assuming this path is correct
# temp_env = SimpleTagEnv(**ENV_CONFIG)
# ALGO_CONFIG = get_maddpg_config("SimpleTagCustomRegistered-v0", temp_env)
# temp_env.close()