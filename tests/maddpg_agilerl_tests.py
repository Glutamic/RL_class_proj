import numpy as np
import torch
from pettingzoo.mpe import simple_tag_v3
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import create_population
from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv
from agilerl.training.train_multi_agent_off_policy import train_multi_agent_off_policy # Correct import

# Type hinting for clarity (optional, but good practice)
from agilerl.training.train_multi_agent_off_policy import PopulationType, InitDictType # From function signature
from typing import List, Optional, Tuple # From function signature

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"===== Using {device} for training =====")

# --- 1. 定义网络配置 (Network Configuration) ---
# This NET_CONFIG is used by create_population, not by train_multi_agent_off_policy directly
NET_CONFIG_ACTOR_CRITIC = { # More explicit naming for clarity when debugging INIT_HP
    "head_config": {"hidden_size": [32, 32]}  # Actor head hidden size
}

# --- 2. 定义初始超参数 (Initial Hyperparameters) ---
INIT_HP: InitDictType = { # Use type hint
    "CHANNELS_LAST": False,
    "BATCH_SIZE": 256,   # 1024
    "O_U_NOISE": True,
    "EXPL_NOISE": 0.1,
    "MEAN_NOISE": 0.0,
    "THETA": 0.15,
    "DT": 0.01,
    "LR_ACTOR": 0.001,
    "LR_CRITIC": 0.001,
    "GAMMA": 0.99,
    "MEMORY_SIZE": 100000,
    "LEARN_STEP": 20,   # 5
    "TAU": 0.01,
    "POLICY_FREQ": 2,
    "POP_SIZE": 2,
    "ALGO": "MADDPG",
    # NET_CONFIG for actor and critic will be implicitly handled by create_population
    # if NET_CONFIG_ACTOR_CRITIC is passed within INIT_HP to create_population,
    # or if create_population has specific args for net_config_actor/critic.
    # For simplicity, we pass NET_CONFIG_ACTOR_CRITIC to create_population directly.
}

num_envs = 32
ENV_NAME = 'simple_tag_v3_agilerl'

# --- 3. 创建向量化的 PettingZoo 环境 ---
env_fns = [
    lambda: simple_tag_v3.parallel_env(max_cycles=100, continuous_actions=True)
    for _ in range(num_envs)
]
vec_env = AsyncPettingZooVecEnv(env_fns)

# --- 4. 配置多智能体算法的输入参数 (for create_population) ---
observation_spaces = [vec_env.single_observation_space(agent) for agent in vec_env.agents]
action_spaces = [vec_env.single_action_space(agent) for agent in vec_env.agents]

# Add N_AGENTS and AGENT_IDS to INIT_HP as create_population might expect them there too
# or pass them directly to create_population.
# The create_population function signature should be checked.
# Let's assume create_population can take these directly, or they are part of INIT_HP.
# For clarity, we ensure they are in INIT_HP if create_population uses it.
INIT_HP["N_AGENTS"] = vec_env.num_agents
INIT_HP["AGENT_IDS"] = vec_env.agents

# --- 5. 创建智能体种群 ---
pop: PopulationType = create_population( # Use type hint
    INIT_HP["ALGO"],
    observation_spaces,
    action_spaces,
    net_config=NET_CONFIG_ACTOR_CRITIC, # Pass the network config here
    INIT_HP=INIT_HP,
    population_size=INIT_HP["POP_SIZE"],
    num_envs=num_envs, # Pass num_envs here
    device=device,
)

# --- 6. 配置多智能体经验回放缓冲区 ---
field_names = ["state", "action", "reward", "next_state", "done"]
memory = MultiAgentReplayBuffer(
    memory_size=INIT_HP["MEMORY_SIZE"],
    field_names=field_names,
    agent_ids=INIT_HP["AGENT_IDS"],
    device=device,
)

# --- 7. 实例化锦标赛选择和变异对象 (用于 HPO) ---
tournament: Optional[TournamentSelection] = TournamentSelection( # Use type hint
    tournament_size=2,
    elitism=True,
    population_size=INIT_HP["POP_SIZE"],
    eval_loop=1, # This eval_loop is for the tournament, train_multi_agent_off_policy also has one.
)

mutations: Optional[Mutations] = Mutations( # Use type hint
    no_mutation=0.2,
    architecture=0.0, # Set to 0 if not mutating architecture with simple MLP
    new_layer_prob=0.0, # Set to 0 if not mutating architecture
    parameters=0.2,
    activation=0.0,
    rl_hp=0.2, # Mutate RL hyperparameters from INIT_HP
    mutation_sd=0.1,
    rand_seed=1,
    device=device,
)

# --- 8. 定义传递给 train_multi_agent_off_policy 的参数 ---
max_train_steps = 10000   # 200000
evo_period_steps = 3000 # 50000 More reasonable evolution frequency than default 25
eval_episodes = 1 # 3 Number of episodes to run for evaluation
learning_delay_steps = INIT_HP["BATCH_SIZE"] * 5
target_training_score: Optional[float] = None # No specific target score for early stopping

# Checkpointing and elite saving paths (optional)
checkpoint_save_path = f"./checkpoints/{ENV_NAME}_{INIT_HP['ALGO']}"
elite_agent_path = f"./elite_agent_{ENV_NAME}_{INIT_HP['ALGO']}.pt"


print(f"===== Starting Training using train_multi_agent_off_policy for {ENV_NAME} =====")

# --- 9. 调用高级训练函数 ---
# Note: MUT_P is for mutation parameters, if not using a pre-configured 'mutation' object.
# Since we pass 'mutations', MUT_P can be None.
trained_pop_final: PopulationType
fitnesses_final: List[List[float]]
trained_pop_final, fitnesses_final = train_multi_agent_off_policy(
    env=vec_env,
    env_name=ENV_NAME,
    algo=INIT_HP["ALGO"],
    pop=pop,
    memory=memory,
    sum_scores=True,  # For simple_tag, sum of rewards might be a proxy for pursuer team performance.
                      # Individual agents in MADDPG still learn from their own rewards.
    INIT_HP=INIT_HP,
    MUT_P=None,       # Using pre-configured 'mutations' object
    swap_channels=INIT_HP['CHANNELS_LAST'], # Which is False for simple_tag_v3
    max_steps=max_train_steps,
    evo_steps=evo_period_steps,
    eval_steps=None,  # Evaluate until episode done
    eval_loop=eval_episodes,
    learning_delay=learning_delay_steps,
    target=target_training_score,
    tournament=tournament,
    mutation=mutations,
    checkpoint=evo_period_steps,  # Checkpoint every evo_steps (or other frequency)
    checkpoint_path=checkpoint_save_path,
    overwrite_checkpoints=False, # Set to True if you want to overwrite
    save_elite=True, # Save the best performing agent at the end
    elite_path=elite_agent_path,
    wb=False,
    verbose=True,
    accelerator=None, # Not using HuggingFace Accelerate in this basic example
    wandb_api_key="bd9fa016592d0c29f46d4158d4716e3e457bfa42" # No W&B logging in this basic example
)

print("===== Training Complete =====")
if fitnesses_final:
    print("Final Population Fitnesses (last evaluation):")
    for i, fitness_list in enumerate(fitnesses_final):
        if fitness_list: # Check if list is not empty
            print(f"  Agent {i}: {fitness_list[-1]}") # Print last recorded fitness
        else:
            print(f"  Agent {i}: No fitness data")


# --- 10. 处理训练后的种群 ---
# The 'trained_pop_final' is the population after training.
# If 'save_elite' was True, the best agent is already saved to 'elite_path'.
# We can also manually find the best agent from 'fitnesses_final' if needed.
if trained_pop_final and fitnesses_final:
    # Find the agent with the highest last fitness score
    last_fitnesses = [fit_list[-1] if fit_list else -float('inf') for fit_list in fitnesses_final]
    if any(f != -float('inf') for f in last_fitnesses): # Check if any valid fitness
        best_agent_idx = np.argmax(last_fitnesses)
        print(f"\nBest agent in final population (index {best_agent_idx}) had fitness: {last_fitnesses[best_agent_idx]:.2f}")
        # The elite agent is already saved if save_elite=True and elite_path is provided.
        # best_agent = trained_pop_final[best_agent_idx]
        # best_agent.save_checkpoint(f"./maddpg_best_overall_{INIT_HP['ALGO']}_{ENV_NAME}.pt")
    else:
        print("No valid fitness data to determine the best agent from the final population.")
# elif save_elite and elite_path:
#     print(f"Elite agent was saved to: {elite_path}")
# else:
#     print("Training finished, but no elite agent explicitly saved via this script's logic (check 'save_elite' parameter).")

vec_env.close()