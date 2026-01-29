import random
import os
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import collections
import mujoco 
from UR5E_rl import UR5E_Env

# === 脚本专家（用于生成演示数据） ===
# 使用状态机和逆运动学 (IK) 来控制机械臂完成抓取任务
class ScriptedExpert:
    def __init__(self, env):
        self.env = env
        self.model = env.model
        self.data = env.data
        self.ee_id = env.ee_id
        # 阶段控制: 0:接近, 1:下降, 2:抓取, 3:抬起, 4:移动到目标
        self.phase = 0 
        self.cnt = 0

    def get_action(self, obs):
        ee_pos = self.data.site_xpos[self.ee_id] # 末端执行器位置
        obj_pos = self.data.xpos[self.env.obj_body_id] # 物体位置
        target_pos = self.env.target_pos # 目标放置位置
        
        desired_pos = list(ee_pos)
        gripper_action = 1.0 # 默认张开爪子
        
        # 计算距离
        dist_xy = np.linalg.norm(ee_pos[:2] - obj_pos[:2])
        dist_z = abs(ee_pos[2] - obj_pos[2])
        
        # 状态机逻辑
        if self.phase == 0: # 阶段0: 水平接近物体上方
            desired_pos[0] = obj_pos[0]
            desired_pos[1] = obj_pos[1]
            desired_pos[2] = obj_pos[2] + 0.20 # 保持在物体上方 20cm
            if dist_xy < 0.02: self.phase = 1 # 如果水平误差小于 2cm，进入下一阶段
            
        elif self.phase == 1: # 阶段1: 垂直下降
            desired_pos = list(obj_pos)
            if dist_z < 0.02: self.phase = 2 # 如果高度误差小于 2cm，进入下一阶段
            
        elif self.phase == 2: # 阶段2: 闭合爪子 (抓取)
            desired_pos = list(obj_pos)
            gripper_action = -1.0 # 闭合信号
            self.cnt += 1
            if self.cnt > 20: # 等待 20 帧确保抓紧
                self.phase = 3
                self.cnt = 0
            
        elif self.phase == 3: # 阶段3: 抬起物体
            desired_pos = list(obj_pos)
            desired_pos[2] = 0.55 # 抬高到 0.55m
            gripper_action = -1.0
            if ee_pos[2] > 0.48: self.phase = 4 # 抬起足够高度后，进入下一阶段
            
        elif self.phase == 4: # 阶段4: 运输到目标点
            desired_pos = list(target_pos)
            gripper_action = -1.0

        # IK 控制 (雅可比矩阵逆运动学)
        error = np.array(desired_pos) - ee_pos
        max_vel = 0.5 # 限制最大速度
        if np.linalg.norm(error) > max_vel:
            error = error / np.linalg.norm(error) * max_vel
            
        jac = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jac, None, self.ee_id)
        jac_arm = jac[:, :6] # 只取前6个关节 (手臂)
        dq = np.linalg.pinv(jac_arm) @ error # 计算关节速度
        
        action = dq / 0.02 # 简单的 P 控制
        action = np.clip(action, -1.0, 1.0)
        final_action = np.append(action, gripper_action)
        return final_action

    def reset(self):
        self.phase = 0
        self.cnt = 0

# === 经验回放池 ===
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)

# === 策略网络 (Actor) ===
# 输入状态，输出动作
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.tanh(self.fc4(x)) * self.action_bound # 输出范围 [-action_bound, action_bound]

# === 价值网络 (Critic) ===
# 输入状态和动作，输出 Q 值
class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        # Critic 输入是 State + Action
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1) 
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc_out(x)

# === TD3 算法主体 ===
class TD3:
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, action_noise,
                 actor_lr, critic_lr, tau, gamma, update_delay, device):
        # 初始化 Actor 和两个 Critic 网络 (双 Q 学习)
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic_1 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic_2 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        
        # 初始化目标网络 (Target Networks)
        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.target_critic_1 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_2 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        
        # 同步初始权重
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        
        self.gamma = gamma # 折扣因子
        self.action_bound = action_bound  
        self.action_noise = action_noise  # 动作探索噪声
        self.tau = tau  # 软更新系数
        self.action_dim = action_dim
        self.update_delay = update_delay # 延迟更新 Actor 的频率
        self.update_iteration = 0      
        self.device = device

    # 选择动作 (测试/推理用)
    def take_action(self, state):
        self.actor.eval()
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        action = self.actor(state)
        # 加上高斯噪声增加探索
        noise = torch.tensor(np.random.normal(loc=0.0, scale=self.action_noise), dtype=torch.float).to(self.device)
        action = torch.clamp(action + noise, -self.action_bound, self.action_bound)
        return action.squeeze().detach().cpu().numpy()

    # 软更新目标网络参数
    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    # 训练更新
    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        with torch.no_grad():
            # 计算目标 Action (加噪声平滑)
            next_actions = self.target_actor(next_states)
            # 计算目标 Q 值 (取两个 Critic 的最小值，缓解过估计)
            next_q_values_1 = self.target_critic_1(next_states, next_actions)
            next_q_values_2 = self.target_critic_2(next_states, next_actions)
            q_targets = rewards + self.gamma * torch.min(next_q_values_1, next_q_values_2) * (1 - dones)

        # 更新两个 Critic
        # critic_1_loss = torch.mean(F.mse_loss(self.critic_1(states, actions), q_targets))
        critic_1_loss = F.smooth_l1_loss(self.critic_1(states, actions), q_targets)
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        #critic_2_loss = torch.mean(F.mse_loss(self.critic_2(states, actions), q_targets))
        critic_2_loss = F.smooth_l1_loss(self.critic_2(states, actions), q_targets)
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 延迟更新 Actor
        self.update_iteration += 1
        if self.update_iteration % self.update_delay == 0:
            actor_loss = -torch.mean(self.critic_1(states, self.actor(states)))
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 软更新目标网络
            self.soft_update(self.actor, self.target_actor)
            self.soft_update(self.critic_1, self.target_critic_1)
            self.soft_update(self.critic_2, self.target_critic_2)

# === 训练主循环 ===
def train_off_policy_agent_parallel(env_list, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    num_envs = len(env_list)
    states = []
    for env in env_list:
        s, _ = env.reset()
        states.append(s)
        
    current_returns = [0] * num_envs
    total_episodes_done = 0
    smooth_return = 0 # 平滑后的回报 (用于显示)
    return_list = []

    with tqdm(total=num_episodes, desc='训练/Training') as pbar:
        while total_episodes_done < num_episodes:
            actions = []
            for state in states:
                actions.append(agent.take_action(state))
            
            for i, env in enumerate(env_list):
                next_state, reward, terminated, truncated, _ = env.step(actions[i])
                current_returns[i] += reward
                
                # 渲染第一个环境
                if i == 0 and hasattr(env.unwrapped, 'viewer') and env.unwrapped.viewer.is_running():
                    env.render()
                
                done = terminated or truncated
                replay_buffer.add(states[i], actions[i], reward, next_state, done)
                states[i] = next_state
                
                if done:
                    states[i], _ = env.reset()
                    if i == 0:
                        total_episodes_done += 1
                        
                        # 计算平滑回报
                        if smooth_return == 0:
                            smooth_return = current_returns[i]
                        else:
                            smooth_return = 0.9 * smooth_return + 0.1 * current_returns[i]
                            
                        # 探索噪声衰减: 让后期动作更精准
                        # if agent.action_noise > 0.02:
                        #     agent.action_noise *= 0.9995

                        pbar.update(1)
                        pbar.set_postfix({
                            'ep': '%d' % total_episodes_done, 
                            'ret': '%.2f' % smooth_return,
                            'noise': '%.3f' % agent.action_noise
                        })
                    return_list.append(current_returns[i])
                    current_returns[i] = 0

            # 更新网络
            if replay_buffer.size() > minimal_size:
                # 每次step 更新多次，加快从专家数据中学习
                for _ in range(num_envs): 
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                                       'dones': b_d}
                    agent.update(transition_dict)
                    
    return return_list

# === 超参数配置 ===
num_envs = 1 # 并行环境数量
env_list = [UR5E_Env() for _ in range(num_envs)]
env = env_list[0] 

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

actor_lr = 1e-4 # Actor 学习率
critic_lr = 1e-3 # Critic 学习率
num_episodes = 2500 # 总训练回合数
hidden_dim = 512 # 隐藏层大小
gamma = 0.98 # 折扣因子
tau = 0.005  # 软更新系数
buffer_size = 1000000 # 经验池大小
minimal_size = 2000 # 开始训练的最小数据量
batch_size = 512 # 批量大小
action_noise = 0.1 # 初始探索噪声
update_delay = 1 # 延迟更新频率
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

replay_buffer = ReplayBuffer(buffer_size)
agent = TD3(state_dim, hidden_dim, action_dim, action_bound, action_noise, actor_lr, critic_lr,
            tau, gamma, update_delay, device)

# === 加载已有模型(如果有) ===
if os.path.exists("TD3_UR5E_Actor.pth"):
    try:
        agent.actor.load_state_dict(torch.load("TD3_UR5E_Actor.pth", map_location=torch.device("cpu")))
    except: pass
if os.path.exists("TD3_UR5E_Critic1.pth"):
    try:
        agent.critic_1.load_state_dict(torch.load("TD3_UR5E_Critic1.pth", map_location=torch.device("cpu")))
    except: pass
if os.path.exists("TD3_UR5E_critic_2.pth"):
    try:
        agent.critic_2.load_state_dict(torch.load("TD3_UR5E_critic_2.pth", map_location=torch.device("cpu")))
    except: pass

# 初始化渲染器
env.unwrapped.viewer_init()

# === 收集专家演示数据 (Imitation Learning) ===
print("收集专家演示数据 (Collecting Expert Demos)...")
expert = ScriptedExpert(env)
demo_episodes = 400 

for i_episode in tqdm(range(demo_episodes), desc='Expert Demo'):
    state, _ = env.reset()
    expert.reset()
    done = False
    
    while not done:
        action = expert.get_action(state)
        # 专家数据也存入 ReplayBuffer
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state
        if i_episode < 5: env.render() # 只渲染前5个演示

# === 预训练 (Pre-training) ===
# 利用专家数据先让 Agent 学一波，避免一开始仅仅是随机乱动
print("正在进行预训练 (Pre-training)...")
pre_train_steps = 4000 # 预训练次数
for _ in tqdm(range(pre_train_steps), desc='Pre-train'):
    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
    transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
    agent.update(transition_dict)

# === 开始正式训练 ===
return_list = train_off_policy_agent_parallel(env_list, agent, num_episodes, replay_buffer, minimal_size, batch_size)

# 保存模型
torch.save(agent.actor.state_dict(), 'TD3_UR5E_Actor.pth')
torch.save(agent.critic_1.state_dict(), 'TD3_UR5E_Critic1.pth')
torch.save(agent.critic_2.state_dict(), 'TD3_UR5E_critic_2.pth')

# 绘制学习曲线
episodes_list = list(range(len(return_list)))
plt.figure()
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('TD3 on UR5E')
plt.savefig('learning_curve.png')