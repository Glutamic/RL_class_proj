import numpy as np
import torch
from mpe2 import simple_tag_v3
from tqdm import trange
import copy
from agilerl.algorithms import MATD3
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv
from agilerl.utils.algo_utils import obs_channels_to_first
import datetime
import os # 导入 os 模块来创建文件夹
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_envs = 4
num_good = 1
num_adversaries = 1
num_obstacles = 1
max_cycles=25
env = AsyncPettingZooVecEnv(
    [
        lambda: simple_tag_v3.parallel_env(num_good=num_good, num_adversaries=num_adversaries, num_obstacles=num_obstacles, max_cycles=max_cycles, continuous_actions=False)
        for _ in range(num_envs)
    ]
)
env.reset()
possible_agent_list = env.agents
print(f"智能体ID列表 (Agent IDs): {possible_agent_list}")
# Configure the multi-agent algo input arguments
observation_spaces = [env.single_observation_space(agent) for agent in env.agents]
action_spaces = [env.single_action_space(agent) for agent in env.agents]

channels_last = False  # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
n_agents = env.num_agents
agent_ids = [agent_id for agent_id in env.agents]


field_names = ["state", "action", "reward", "next_state", "done"]
memory = MultiAgentReplayBuffer(
    100_000,
    field_names=field_names,
    agent_ids=agent_ids,
    device=device,
)

NET_CONFIG = {
        "encoder_config": {
            "hidden_size": [128],
        },
        "head_config": {
            "hidden_size": [128],
        },
    }

agent = MATD3(
    observation_spaces=observation_spaces,
    action_spaces=action_spaces,
    agent_ids=agent_ids,
    net_config=NET_CONFIG,
    batch_size=1024,         # 每次学习时采样的经验数量
    vect_noise_dim=num_envs,
    learn_step=32,           # 学习间隔，分类讨论大于环境数和小于环境数的情况，最好是环境数的整数倍或者反之
    device=device,
)
print(f"MADDPG 智能体已初始化。使用的设备: {agent.device}")

# Define training loop parameters
max_steps = 50000  # Max steps
total_steps = 0
training_steps = 10000  # Frequency at which we evaluate training score
warmup_steps = 3000   # 在开始学习前，先用随机/探索策略填充经验池的步数
eval_steps = None  # Evaluation steps per episode - go until done
eval_loop = 1  # Number of evaluation episodes

# TRAINING LOOP
print(f"\n开始训练，共 {max_steps} 步...")
pbar = trange(max_steps, unit="step")
while np.less(agent.steps[-1], max_steps):
    steps = 0
    state, info  = env.reset() # Reset environment at start of episode
    scores = np.zeros((num_envs, len(agent_ids)))
    completed_episode_scores = []
    if channels_last:
        state = {agent_id: obs_channels_to_first(s) for agent_id, s in state.items()}

    for idx_step in range(training_steps // num_envs):
        # Get next action from agent
        cont_actions, discrete_action = agent.get_action(
            state,
            training=True,
            infos=info,
        )
        if agent.discrete_actions:
            action = discrete_action
        else:
            action = cont_actions

        # Use random action to train
        # action = {agent: env.action_space(agent).sample() for agent in env.agents}
        # Act in environment
        next_state, reward, termination, truncation, info = env.step(action)
        # print("agent reward", reward)

        scores += np.array(list(reward.values())).transpose()
        # print(scores)
        total_steps += num_envs
        steps += num_envs

        # Save experiences to replay buffer
        if channels_last:
            next_state = {
                agent_id: obs_channels_to_first(ns)
                for agent_id, ns in next_state.items()
            }

        memory.save_to_memory(state, cont_actions, reward, next_state, termination, is_vectorised=True)
        # 如果学习间隔比并行的环境数量多，那么学习间隔减小到num_envs分之一，但疑似也会有不均匀的情况，值得注意（idx_step归零）
        if agent.learn_step > num_envs:
            learn_step = agent.learn_step // num_envs
            if (
                idx_step % learn_step == 0
                and len(memory) >= agent.batch_size
                and memory.counter > warmup_steps
            ):
                # Sample replay buffer
                experiences = memory.sample(agent.batch_size)
                # Learn according to agent's RL algorithm
                agent.learn(experiences)

        # Handle num_envs > learn step; learn multiple times per step in env
        elif len(memory) >= agent.batch_size and memory.counter > warmup_steps:
            for _ in range(num_envs // agent.learn_step):
                # Sample replay buffer
                experiences = memory.sample(agent.batch_size)
                # Learn according to agent's RL algorithm
                agent.learn(experiences)

        # Update the state
        state = next_state

        # Calculate scores and reset noise for finished episodes
        reset_noise_indices = []
        term_array = np.array(list(termination.values())).transpose()
        trunc_array = np.array(list(truncation.values())).transpose()
        for idx, (d, t) in enumerate(zip(term_array, trunc_array)):
            if np.any(d) or np.any(t):
                completed_episode_scores.append(copy.deepcopy(scores[idx]))
                agent.scores.append(copy.deepcopy(scores[idx]))
                scores[idx] = 0
                reset_noise_indices.append(idx)
        agent.reset_action_noise(reset_noise_indices)

    pbar.update(training_steps)
    agent.steps[-1] += steps

    # Evaluate population
    fitness = agent.test(
        env,
        swap_channels=False,
        max_steps=eval_steps,
        loop=eval_loop,
        sum_scores=False,
    )
    pop_episode_scores = np.array(completed_episode_scores)
    mean_scores = np.mean(pop_episode_scores, axis=0)

    print(f"--- Global steps {total_steps} ---")
    print(f"Steps {agent.steps[-1]}")
    print("Scores:")
    for idx, sub_agent in enumerate(agent_ids):
        print(f"    {sub_agent} score: {mean_scores[idx]}")
    print("Fitness")
    for idx, sub_agent in enumerate(agent_ids):
        print(f"    {sub_agent} fitness: {fitness[idx]}")
    print("Previous 5 fitness avgs")
    for idx, sub_agent in enumerate(agent_ids):
        print(
            f"  {sub_agent} fitness average: {np.mean(agent.fitness[-5:], axis=0)[idx]}"
        )

    # Update step counter
    agent.steps.append(agent.steps[-1])

# Save the trained algorithm
elite_base_path = f"./elite_agents/simple_tag_v3_agilerl_MATD3"

# 结合时间戳创建具体的保存路径
elite_agent_save_path = f"{elite_base_path}/{current_time}/" 
elite_agent_filename = f"elite_agent_simple_tag_v3_agilerl_MATD3_{num_good}_{num_adversaries}_{num_obstacles}_.pt"

if not os.path.exists(elite_agent_save_path):
    os.makedirs(elite_agent_save_path)

elite_agent_path = os.path.join(elite_agent_save_path, elite_agent_filename)
agent.save_checkpoint(elite_agent_path)

pbar.close()
env.close()
