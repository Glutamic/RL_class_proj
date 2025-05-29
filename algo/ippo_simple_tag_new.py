import numpy as np
import torch
from pettingzoo.mpe import simple_tag_v3
from tqdm import trange
from gymnasium import spaces
from agilerl.algorithms import IPPO
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv
from agilerl.utils.algo_utils import obs_channels_to_first

import datetime
import os # 导入 os 模块来创建文件夹
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_envs = 4
env = AsyncPettingZooVecEnv(
    [
        lambda: simple_tag_v3.parallel_env(num_adversaries=1, num_obstacles=1, continuous_actions=True)
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
    1_000_000,
    field_names=field_names,
    agent_ids=agent_ids,
    device=device,
)

NET_CONFIG = {
        "encoder_config": {
            "hidden_size": [128, 128],
        },
        "head_config": {
            "hidden_size": [128, 64],
        },
    }

agent = IPPO(
    observation_spaces=observation_spaces,
    action_spaces=action_spaces,
    agent_ids=agent_ids,
    net_config=NET_CONFIG,
    device=device,
)
print(f"IPPO 智能体已初始化。使用的设备: {agent.device}")

# Define training loop parameters
max_steps = 2000000  # Max steps
total_steps = 0
training_steps = 10000  # Frequency at which we evaluate training score
warmup_steps = 500   # 在开始学习前，先用随机/探索策略填充经验池的步数
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

    for idx_step in range(training_steps // num_envs):
        # Get next action from agent
        states = {agent_id: [] for agent_id in agent.agent_ids}
        actions = {agent_id: [] for agent_id in agent.agent_ids}
        log_probs = {agent_id: [] for agent_id in agent.agent_ids}
        rewards = {agent_id: [] for agent_id in agent.agent_ids}
        dones = {agent_id: [] for agent_id in agent.agent_ids}
        values = {agent_id: [] for agent_id in agent.agent_ids}

        done = {agent_id: np.zeros(num_envs) for agent_id in agent.agent_ids}

        for idx_step in range(-(agent.learn_step // -num_envs)):

            # Get next action from agent
            action, log_prob, _, value = agent.get_action(obs=state, infos=info)

            # Clip to action space
            clipped_action = {}
            for agent_id, agent_action in action.items():
                shared_id = agent.get_homo_id(agent_id)
                actor_idx = agent.shared_agent_ids.index(shared_id)
                agent_space = agent.action_space[agent_id]
                if isinstance(agent_space, spaces.Box):
                    if agent.actors[actor_idx].squash_output:
                        clipped_agent_action = agent.actors[actor_idx].scale_action(agent_action)
                    else:
                        clipped_agent_action = np.clip(agent_action, agent_space.low, agent_space.high)
                else:
                    clipped_agent_action = agent_action

                clipped_action[agent_id] = clipped_agent_action

            # Act in environment
            next_state, reward, termination, truncation, info = env.step(clipped_action)
            scores += np.array(list(reward.values())).transpose()

            steps += num_envs

            next_done = {}
            for agent_id in agent.agent_ids:
                states[agent_id].append(state[agent_id])
                actions[agent_id].append(action[agent_id])
                log_probs[agent_id].append(log_prob[agent_id])
                rewards[agent_id].append(reward[agent_id])
                dones[agent_id].append(done[agent_id])
                values[agent_id].append(value[agent_id])
                next_done[agent_id] = np.logical_or(termination[agent_id], truncation[agent_id]).astype(np.int8)

            if channels_last:
                next_state = {
                    agent_id: obs_channels_to_first(s)
                    for agent_id, s in next_state.items()
                }

            # Find which agents are "done" - i.e. terminated or truncated
            finished = {
                agent_id: termination[agent_id] | truncation[agent_id]
                for agent_id in agent.agent_ids
            }

            # Calculate scores for completed episodes
            for idx, agent_dones in enumerate(zip(*finished.values())):
                if all(agent_dones):
                    completed_score = list(scores[idx])
                    completed_episode_scores.append(completed_score)
                    agent.scores.append(completed_score)
                    scores[idx].fill(0)

            state = next_state
            done = next_done

        experiences = (
            states,
            actions,
            log_probs,
            rewards,
            dones,
            values,
            next_state,
            next_done,
        )

        # Learn according to agent's RL algorithm
        loss = agent.learn(experiences)

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
elite_base_path = f"./elite_agents/simple_tag_v3_agilerl_IPPO"

# 结合时间戳创建具体的保存路径
elite_agent_save_path = f"{elite_base_path}/{current_time}/" 
elite_agent_filename = f"elite_agent_simple_tag_v3_agilerl_IPPO_{current_time}.pt"

if not os.path.exists(elite_agent_save_path):
    os.makedirs(elite_agent_save_path)

elite_agent_path = os.path.join(elite_agent_save_path, elite_agent_filename)
agent.save_checkpoint(elite_agent_path)

pbar.close()
env.close()
