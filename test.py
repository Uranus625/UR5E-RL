import numpy as np
from UR5E_rl import UR5E_Env, NormalizationWrapper
import time
import torch
import torch.nn.functional as F
import os
import mujoco
import xml.etree.ElementTree as ET

# === 动态创建带有障碍物的场景帮助函数 ===
def create_scene_with_distractors(original_path, new_path, num_distractors=8):
    # 解析原始 XML
    tree = ET.parse(original_path)
    root = tree.getroot()
    worldbody = root.find('worldbody')
    
    # 定义障碍物颜色 (绿, 蓝, 黄, 青, 洋红)
    colors = ["0 1 0 1", "0 0 1 1", "1 1 0 1", "0 1 1 1", "1 0 1 1"] 
    
    for i in range(num_distractors):
        color = colors[i % len(colors)]
        body = ET.SubElement(worldbody, 'body')
        body.set('name', f'distractor{i}')
        body.set('pos', f'0.38 {0.1 + i*0.05} 0.45') # 初始位置
        
        joint = ET.SubElement(body, 'freejoint')
        joint.set('name', f'distractor{i}_joint')
        
        geom = ET.SubElement(body, 'geom')
        geom.set('name', f'distractor{i}_geom')
        geom.set('type', 'box')
        geom.set('size', '0.02 0.02 0.02')
        geom.set('rgba', color)
        geom.set('mass', '0.1')
        geom.set('contype', '1')
        geom.set('conaffinity', '1')
        geom.set('friction', '1 0.5 0.5')

    tree.write(new_path)

# === 扩展环境类 ===
class DistractorUR5E_Env(UR5E_Env):
    def __init__(self):
        # 1. 生成修改后的 XML
        original_xml = 'model/universal_robots_ur5e/scene.xml'
        self.temp_xml = 'model/universal_robots_ur5e/scene_distractors_temp.xml'
        self.num_distractors = 8 # 障碍物数量
        create_scene_with_distractors(original_xml, self.temp_xml, self.num_distractors)
        
        # 2. 初始化父类 (加载原始模型)
        super().__init__()
        
        # 3. 重新加载带有障碍物的模型
        self.model = mujoco.MjModel.from_xml_path(self.temp_xml)
        self.data = mujoco.MjData(self.model)
        
        # 4. 重新初始化 IDs (复制自 UR5E_rl.py，因为模型改变，ID 需要重新获取)
        self.ee_id = self.model.site("attachment_site").id
        self.obj_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_joint")
        self.obj_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        self.obj_qpos_adr = self.model.jnt_qposadr[self.obj_joint_id]
        self.right_finger_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_finger")
        self.left_finger_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_finger")
        self.object_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
        self.table_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "table")
        
        # 5. 获取障碍物关节地址
        self.distractor_qpos_adrs = []
        for i in range(self.num_distractors):
            j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f'distractor{i}_joint')
            if j_id != -1:
                self.distractor_qpos_adrs.append(self.model.jnt_qposadr[j_id])

    def reset(self, **kwargs):
        # 调用父类 reset
        obs, info = super().reset(**kwargs)
        
        # 随机化障碍物位置
        for adr in self.distractor_qpos_adrs:
            # 随机在桌面上生成 (靠近物体)
            dx = 0.38 + np.random.uniform(-0.15, 0.15)
            # 只能生成在左边 (Y > 0)
            dy = np.random.uniform(0.0, 0.3)
            dz = 0.45 
            self.data.qpos[adr:adr+3] = [dx, dy, dz]
            self.data.qpos[adr+3:adr+7] = [1, 0, 0, 0]

        mujoco.mj_step(self.model, self.data) # 应用更改
        return self._get_obs(), info

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim) # 增加层
        self.fc4 = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x)) # 增加激活函数
        return torch.tanh(self.fc4(x)) * self.action_bound

class TD3_test():
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.device = device

    def take_action(self, state):
        self.actor.eval()
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        action = self.actor(state)
        return action.squeeze().detach().cpu().numpy()


# 使用扩展环境
try:
    env = DistractorUR5E_Env()
    print("使用了带有障碍物的环境")
except Exception as e:
    print(f"加载障碍物失败: {e}")
    env = UR5E_Env()

action_bound = env.action_space.high[0]
action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]
hidden_dim = 512
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
agent_test = TD3_test(state_dim, hidden_dim, action_dim, action_bound, device)
agent_test.actor.load_state_dict(torch.load("TD3_UR5E_Actor.pth", map_location=torch.device("cpu")))

env.unwrapped.viewer_init()
while env.unwrapped.viewer.is_running():
    for episode in range(10):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent_test.take_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            env.render()
            time.sleep(0.01)
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

