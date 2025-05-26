import os
import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

from agilerl.algorithms.maddpg import MADDPG # 导入 MADDPG 算法
from pettingzoo.mpe import simple_tag_v3 # 更改为 simple_tag_v3 环境

# 定义在帧上标注回合数的函数
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

    # 网络配置，应与训练时使用的配置一致
    NET_CONFIG_ACTOR_CRITIC = {
        "head_config": {"hidden_size": [32, 32]}
    }

    # 配置 simple_tag_v3 环境
    env = simple_tag_v3.parallel_env(
        render_mode="rgb_array", continuous_actions=True, max_cycles=100
    )
    
    # simple_tag_v3 使用向量观测，不需要图像相关的预处理
    
    env.reset() # 初始化环境以获取智能体和空间信息

    # 获取观测空间和动作空间列表
    # 注意：这里传递给 MADDPG 的是 gym.spaces 对象列表
    observation_spaces = [env.observation_space(agent) for agent in env.agents]
    action_spaces = [env.action_space(agent) for agent in env.agents]

    # 智能体数量和 ID
    n_agents = env.num_agents
    agent_ids = env.agents

    # 实例化 MADDPG 对象
    # 严格按照你提供的 space_invaders_v2 示例的参数风格
    # 假设 MADDPG 类能从 action_spaces (Box类型) 中正确推断出连续动作及相关边界
    maddpg = MADDPG(
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        agent_ids=agent_ids,
        net_config=NET_CONFIG_ACTOR_CRITIC,  # 传递与训练时一致的网络配置
        device=device,
    )

    # 请将此路径替换为你实际训练好的 simple_tag_v3 模型文件的路径
    path_to_simple_tag_model = "./elite_agents/simple_tag_v3_agilerl_MADDPG/20250521_094535/elite_agent_simple_tag_v3_agilerl_MADDPG_20250521_094535.pt" # <-- 更新此路径
    
    if not os.path.exists(path_to_simple_tag_model) or "YOUR_ELITE_MODEL.pt" in path_to_simple_tag_model:
        print(f"警告: 模型路径 {path_to_simple_tag_model} 似乎是占位符或不存在。")
        print("请将 'path_to_simple_tag_model' 更新为你训练好的智能体的正确路径。")
        # 如果路径无效，可以选择退出 exit() 
    
    try:
        maddpg.load_checkpoint(path_to_simple_tag_model)
        print(f"成功从以下路径加载模型: {path_to_simple_tag_model}")
    except Exception as e:
        print(f"从 {path_to_simple_tag_model} 加载检查点失败: {e}")
        print("请确保路径正确，且模型与当前的 MADDPG 配置兼容。")
        exit()

    # 定义测试循环参数
    episodes = 3  # 测试的回合数
    max_episode_steps = 500 # 每个回合的最大步数使用环境的 max_cycles

    rewards_history = []  # 收集每个回合的总奖励
    frames = []  # 收集帧用于制作 GIF
    individual_agent_rewards_history = {
        agent_id: [] for agent_id in agent_ids
    }  # 收集每个智能体的回合奖励

    # 推理测试循环
    for ep in range(episodes):
        state, info = env.reset() # state 是一个字典: {agent_id: observation}
        current_episode_agent_rewards = {agent_id: 0 for agent_id in agent_ids}
        
        # 确保观测值是 numpy 数组 (PettingZoo 通常返回 numpy 数组)
        for agent_id_key in state:
            if not isinstance(state[agent_id_key], np.ndarray):
                state[agent_id_key] = np.array(state[agent_id_key], dtype=np.float32)

        for step in range(max_episode_steps):
            # simple_tag_v3 的观测是向量，不需要通道顺序处理

            # 从智能体获取下一个动作
            # 对于 MADDPG 的连续动作，training=False (或 epsilon=0) 用于确定性动作
            with torch.no_grad(): # 推理时不需要计算梯度
                # MADDPG.get_action 返回 (cont_actions, discrete_action)
                # cont_actions 是一个字典 {agent_id: numpy_action_array}
                cont_actions, _ = maddpg.get_action(state, training=False)

            # simple_tag_v3 配置为连续动作, maddpg.discrete_actions 应为 False
            # action_to_step 直接使用 cont_actions
            action_to_step = cont_actions

            # 保存当前帧用于制作 GIF
            frame = env.render()
            frames.append(_label_with_episode_number(frame, episode_num=ep))

            # 在环境中执行动作
            action = {agent_id: a[0] for agent_id, a in action_to_step.items()}
            next_state, reward, termination, truncation, info = env.step(action)

            # 更新状态 (并确保是 numpy 数组)
            for agent_id_key in next_state:
                if not isinstance(next_state[agent_id_key], np.ndarray):
                    state[agent_id_key] = np.array(next_state[agent_id_key], dtype=np.float32)
                else:
                    state[agent_id_key] = next_state[agent_id_key]
            
            # 累加当前回合每个智能体的奖励
            for agent_id_key, r_val in reward.items():
                current_episode_agent_rewards[agent_id_key] += r_val

            # 如果所有智能体都终止或回合被截断，则结束当前回合
            if any(termination.values()) or any(truncation.values()):
                # 渲染终止/截断后的最后一帧
                frame = env.render()
                frames.append(_label_with_episode_number(frame, episode_num=ep))
                break
        
        # 计算当前回合的总奖励
        current_episode_score = sum(current_episode_agent_rewards.values())
        rewards_history.append(current_episode_score)

        # 记录每个智能体当前回合的奖励
        for agent_id_key in agent_ids:
            individual_agent_rewards_history[agent_id_key].append(current_episode_agent_rewards[agent_id_key])

        print("-" * 15, f"回合: {ep+1}", "-" * 15)
        print(f"回合总奖励: {rewards_history[-1]:.2f}")
        for agent_id_key, reward_list in individual_agent_rewards_history.items():
            print(f"  智能体 {agent_id_key} 奖励: {reward_list[-1]:.2f}")
        if not (any(termination.values()) or any(truncation.values())): # 检查是否因为达到max_steps而结束
             print(f"警告: 回合 {ep+1} 达到了最大步数 ({max_episode_steps})，但并非所有智能体都发出终止/截断信号。")

    env.close()

    # 保存 GIF 到指定路径
    gif_path = "./videos/"
    os.makedirs(gif_path, exist_ok=True)
    output_gif_filename = "simple_tag_v3_evaluation.gif"
    try:
        imageio.mimwrite(
            os.path.join(gif_path, output_gif_filename), frames, fps=10 # fps 可调整
        )
        print(f"成功保存 GIF 至: {os.path.join(gif_path, output_gif_filename)}")
    except Exception as e:
        print(f"保存 GIF 失败: {e}")