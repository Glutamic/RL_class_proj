from pettingzoo.mpe import simple_tag_v3
from matplotlib import pyplot as plt
from agilerl.algorithms.maddpg import MADDPG # 导入 MADDPG 算法
import os
import imageio
import numpy as np
import supersuit as ss
from PIL import Image, ImageDraw
import torch

def _label_with_episode_number(frame, episode_num):
    im = Image.fromarray(frame)
    drawer = ImageDraw.Draw(im)

    # 根据帧的平均亮度选择文字颜色
    if np.mean(frame) < 128:
        text_color = (255, 255, 255)
    else:
        text_color = (0, 0, 0)
    drawer.text(
        (im.size[0] / 20, im.size[1] / 18), f"回合: {episode_num+1}", fill=text_color
    )
    return im


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = simple_tag_v3.parallel_env(render_mode="rgb_array", max_cycles=100, continuous_actions=False, dynamic_rescaling=True)
    env.reset()
    observation_spaces = [env.observation_space(agent) for agent in env.agents]
    action_spaces = [env.action_space(agent) for agent in env.agents]

    n_agents = env.num_agents
    agent_ids = env.agents

    NET_CONFIG = {
            "encoder_config": {
                "hidden_size": [128, 128],
            },
            "head_config": {
                "hidden_size": [128, 64],
            },
        }

    path = "/home/guchengou/MARL_Project/elite_agents/simple_tag_v3_agilerl_MATD3/elite_agent_simple_tag_v3_agilerl_MADDPG_20250529_004611.pt"
    agent = MADDPG.load(path, device)
    print(f"MADDPG 智能体已初始化。使用的设备: {agent.device}")

    # 定义测试循环参数
    episodes = 3  # 测试的回合数
    max_steps = 500 # 每个回合的最大步数使用环境的 max_cycles

    rewards = []  # List to collect total episodic reward
    frames = []  # 收集帧用于制作 GIF
    indi_agent_rewards = {
        agent_id: [] for agent_id in agent_ids
    }  # 收集每个智能体的回合奖励

    for ep in range(episodes):
        state, info = env.reset()
        agent_reward = {agent_id: 0 for agent_id in agent_ids}
        score = 0
        for _ in range(max_steps):
            # Get next action from agent
            cont_actions, discrete_action = agent.get_action(
                state, training=False, infos=info
            )
            action = discrete_action
            print(action)
            # Save the frame for this step and append to frames list
            frame = env.render()
            frames.append(_label_with_episode_number(frame, episode_num=ep))
            # Take action in environment
            print(action)
            state, reward, termination, truncation, info = env.step(action)

            # Save agent's reward for this step in this episode
            for agent_id, r in reward.items():
                agent_reward[agent_id] += r

            # Determine total score for the episode and then append to rewards list
            score = sum(agent_reward.values())

            # Stop episode if any agents have terminated
            if any(truncation.values()) or any(termination.values()):
                break

        rewards.append(score)

        # Record agent specific episodic reward for each agent
        for agent_id in agent_ids:
            indi_agent_rewards[agent_id].append(agent_reward[agent_id])

        print("-" * 15, f"Episode: {ep}", "-" * 15)
        print("Episodic Reward: ", rewards[-1])
        for agent_id, reward_list in indi_agent_rewards.items():
            print(f"{agent_id} reward: {reward_list[-1]}")
    env.close()

    # Save the gif to specified path
    gif_path = "./videos/"
    os.makedirs(gif_path, exist_ok=True)
    imageio.mimwrite(
        os.path.join("./videos/", "simple_tag_maddpg.gif"), frames, duration=10
    )