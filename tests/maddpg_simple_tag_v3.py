import numpy as np
import torch
from pettingzoo.mpe import simple_tag_v3
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import create_population
from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv
from agilerl.training.train_multi_agent_off_policy import train_multi_agent_off_policy
from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
from agilerl.training.train_multi_agent_off_policy import PopulationType, InitDictType
from typing import List, Optional, Tuple
import wandb
import argparse

# 添加命令行参数解析
def parse_args():
    parser = argparse.ArgumentParser(description='MADDPG training for Simple Tag environment')
    
    # 训练相关参数
    parser.add_argument('--max_train_steps', type=int, default=500000, help='Maximum training steps')
    parser.add_argument('--training_steps', type=int, default=20000, help='Frequency of training evaluation')
    parser.add_argument('--evo_period_steps', type=int, default=25000, help='Evolution period steps')
    parser.add_argument('--eval_episodes', type=int, default=1, help='Number of evaluation episodes')
    parser.add_argument('--learning_delay_steps', type=int, default=4096, help='Learning delay steps')
    
    # 算法超参数
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training')
    parser.add_argument('--lr_actor', type=float, default=0.001, help='Learning rate for actor')
    parser.add_argument('--lr_critic', type=float, default=0.001, help='Learning rate for critic')
    parser.add_argument('--memory_size', type=int, default=100000, help='Size of replay buffer')
    parser.add_argument('--pop_size', type=int, default=2, help='Population size')
    
    # 环境相关
    parser.add_argument('--num_envs', type=int, default=8, help='Number of parallel environments')
    
    # 其他配置
    parser.add_argument('--checkpoint_freq', type=int, default=10000, help='Checkpoint saving frequency')
    parser.add_argument('--use_wandb', action='store_true', help='Whether to use wandb')
    parser.add_argument('--wandb_key', type=str, default="bd9fa016592d0c29f46d4158d4716e3e457bfa42", help='WandB API key')
    
    args = parser.parse_args()
    return args

# 解析命令行参数
args = parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"===== Using {device} for training =====")

# --- 1. 定义网络配置 (Network Configuration) ---
NET_CONFIG_ACTOR_CRITIC = { # More explicit naming for clarity when debugging INIT_HP
        "head_config": {"hidden_size": [64, 32]},  # Actor head hidden size
}

# --- 2. 定义初始超参数 (Initial Hyperparameters) ---
INIT_HP: InitDictType = { # Use type hint
    "CHANNELS_LAST": False,
    "BATCH_SIZE": args.batch_size,   # 1024
    "O_U_NOISE": True,
    "EXPL_NOISE": 0.1,
    "MEAN_NOISE": 0.0,
    "THETA": 0.15,
    "DT": 0.01,
    "LR_ACTOR": args.lr_actor,
    "LR_CRITIC": args.lr_critic,
    "GAMMA": 0.99,
    "MEMORY_SIZE": args.memory_size,
    "LEARN_STEP": 20,   # 5
    "TAU": 0.01,
    "POLICY_FREQ": 2,
    "POP_SIZE": args.pop_size,
    "ALGO": "MADDPG",
}

num_envs = args.num_envs
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

INIT_HP["N_AGENTS"] = vec_env.num_agents
INIT_HP["AGENT_IDS"] = vec_env.agents

hp_config = HyperparameterConfig(
        lr_actor=RLParameter(min=1e-4, max=1e-2),
        lr_critic=RLParameter(min=1e-4, max=1e-2),
        batch_size=RLParameter(min=8, max=512, dtype=int),
        learn_step=RLParameter(
            min=20, max=200, dtype=int, grow_factor=1.5, shrink_factor=0.75
        ),
    )

# --- 5. 创建智能体种群 ---
pop: PopulationType = create_population(
    INIT_HP["ALGO"],
    observation_spaces,
    action_spaces,
    NET_CONFIG_ACTOR_CRITIC, # Pass the network config here
    INIT_HP,
    hp_config,
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
max_train_steps = args.max_train_steps   # 200000
training_steps = args.training_steps  # Frequency at which we evaluate training score
evo_period_steps = args.evo_period_steps #  More reasonable evolution frequency than default 25
eval_episodes = args.eval_episodes # 3 Number of episodes to run for evaluation
learning_delay_steps = args.learning_delay_steps  # INIT_HP["BATCH_SIZE"] * 5
target_training_score: Optional[float] = None # No specific target score for early stopping

import datetime
import os # 导入 os 模块来创建文件夹

# 1. 获取当前时间并格式化
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # 例如: 20250520_203045

# 2. 更新存储路径
checkpoint_base_path = f"./checkpoints/{ENV_NAME}_{INIT_HP['ALGO']}"
elite_base_path = f"./elite_agents/{ENV_NAME}_{INIT_HP['ALGO']}"

# 结合时间戳创建具体的保存路径
checkpoint_save_path = f"{checkpoint_base_path}/{current_time}/"
elite_agent_save_path = f"{elite_base_path}/{current_time}/" 
elite_agent_filename = f"elite_agent_{ENV_NAME}_{INIT_HP['ALGO']}_{current_time}.pt"

if not os.path.exists(checkpoint_save_path):
    os.makedirs(checkpoint_save_path) 

if not os.path.exists(elite_agent_save_path):
    os.makedirs(elite_agent_save_path)

elite_agent_path = os.path.join(elite_agent_save_path, elite_agent_filename)


print(f"===== Starting Training using train_multi_agent_off_policy for {ENV_NAME} =====")

# --- 9. 调用高级训练函数 ---
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
    checkpoint=args.checkpoint_freq,  # Checkpoint every evo_steps (or other frequency)
    checkpoint_path=checkpoint_save_path,
    overwrite_checkpoints=False, # Set to True if you want to overwrite
    save_elite=True, # Save the best performing agent at the end
    elite_path=elite_agent_path,
    wb=args.use_wandb,
    verbose=True,
    accelerator=None, # Not using HuggingFace Accelerate in this basic example
    wandb_api_key=args.wandb_key
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
if trained_pop_final and fitnesses_final:
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