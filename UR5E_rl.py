import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces

# 建立ur5e机械臂强化学习环境
# 任务：控制机械臂末端夹爪捡取桌子上的物体并放置到指定目标位置
class UR5E_Env(gym.Env):
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path('model/universal_robots_ur5e/scene.xml')
        self.data = mujoco.MjData(self.model)
        
        # Initial State: Make robot face the table (Forward)
        # Previous [-1.57...] was facing sideways (-Y). We want roughly +X
        # A good 'home' above table: [0, -1.57, 1.57, -1.57, -1.57, 0]
        # Adjusted to lift elbow: [0, -2.3, 2.0, -1.2, -1.57, 0] (Shoulder back more, Elbow bent more)
        self.start_joints = np.array([0.0, -1.57, 1.0, -1.5, -1.57, 0.0])
        self.ee_id = self.model.site("attachment_site").id
        
        self.obj_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_joint")
        self.obj_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        self.obj_qpos_adr = self.model.jnt_qposadr[self.obj_joint_id]

        # 获取用于碰撞检测的 Geom ID / Get Geom IDs
        self.right_finger_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_finger")
        self.left_finger_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_finger")
        self.object_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
        
        # 桌身碰撞几何体 ID / Table collision geometry
        self.table_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "table")

        # 动作空间：6个关节 + 1个夹爪 / Action: 6 joints + 1 gripper
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        
        # 观测空间 / Observation: 
        # qpos(6) + qvel(6) + gripper(1) + obj_pos(3) + rel_obj(3) + rel_target(3) + ee_z_axis(3) = 25
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32)

        self.iteration = 0
        self.on_goal_count = 0
        self.done = False
        
        # 记录目标关节位置（非物理位置），防止增量控制导致的重力漂移
        # Keep track of the target joint positions to avoid gravity drift
        self.target_joint_pos = np.copy(self.start_joints)

    def viewer_init(self):
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.viewer.cam.lookat[:] = [0.5, 0, 0.4]
        self.viewer.cam.distance = 1.5
        self.viewer.cam.azimuth = 90
        self.viewer.cam.elevation = -30

    def reset(self, seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)
        
        # 重置机器人状态 / Reset robot
        self.data.qpos[:6] = self.start_joints
        self.data.ctrl[:6] = self.start_joints
        self.data.ctrl[6:8] = 0 # 夹爪张开
        self.target_joint_pos = np.copy(self.start_joints) # 重置目标追踪器
        
        # 重置物体位置（随机在桌面上，但在目标点之外）
        # Reset object position (randomly on table, but away from target)
        # Table center is at x=0.38. Range +/- 0.1 to stay safely on table (size 0.15)
        obj_x = 0.38 + np.random.uniform(-0.1, 0.1)
        obj_y = np.random.uniform(0.05, 0.25) 
        obj_z = 0.45 
        
        self.data.qpos[self.obj_qpos_adr:self.obj_qpos_adr+3] = [obj_x, obj_y, obj_z]
        self.data.qpos[self.obj_qpos_adr+3:self.obj_qpos_adr+7] = [1, 0, 0, 0] 
        
        # 重置目标点 / Reset Target
        # 固定目标位置 / Fixed Target Position
        self.target_pos = np.array([0.38, -0.15, 0.45]) # 物体目标中心高度
        # self.model.geom('target_point').pos = self.target_pos 

        mujoco.mj_step(self.model, self.data)

        self.iteration = 0
        self.on_goal_count = 0
        return self._get_obs(), {}

    def _get_obs(self):
        qpos = self.data.qpos[:6]
        qvel = self.data.qvel[:6] # 速度信息
        gripper_width = self.data.qpos[6] + self.data.qpos[7]
        obj_pos = self.data.xpos[self.obj_body_id]
        ee_pos = self.data.site_xpos[self.ee_id]
        
        # 获取末端执行器朝向（Y轴/手指方向）
        # Get End-Effector Orientation (Y-axis / Finger direction)
        site_mat = self.data.site_xmat[self.ee_id].reshape(3, 3)
        ee_y_axis = site_mat[:, 1] # 手指指向的方向

        obs = np.hstack([
            qpos,
            qvel, # 添加速度到观测
            [gripper_width],
            obj_pos,
            obj_pos - ee_pos,
            self.target_pos - obj_pos,
            ee_y_axis # 手指方向向量
        ]).astype(np.float32)
        return obs

    def _check_contact(self):
        touch_left = False
        touch_right = False
        table_collision = False
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geoms = [contact.geom1, contact.geom2]
            
            # 1. 手指与物体接触 / Finger-Object Contact
            if self.object_geom_id in geoms:
                if self.left_finger_geom_id in geoms:
                    touch_left = True
                if self.right_finger_geom_id in geoms:
                    touch_right = True
            
            # 2. 桌子碰撞检测（机器人任何部位触碰桌子）/ Table Collision
            b1 = self.model.geom_bodyid[contact.geom1]
            b2 = self.model.geom_bodyid[contact.geom2]
            if b1 == self.table_body_id or b2 == self.table_body_id:
                # 排除物体与桌子的合法接触，只检测机器人撞桌子
                is_obj = (contact.geom1 == self.object_geom_id or contact.geom2 == self.object_geom_id)
                if not is_obj:
                     table_collision = True

        return (touch_left or touch_right), table_collision

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        
        # 机器人控制 / Robot Control
        # 减小动作缩放比例以降低速度和顿挫感
        # Decrease action scaling to reduce speed and jerkiness
        self.target_joint_pos += action[:6] * 0.02
        
        # 应用关节限制（限制工作空间） / Apply Joint Limits
        # Base: +/- 90 degrees around front
        self.target_joint_pos[0] = np.clip(self.target_joint_pos[0], -1.57, 1.57) 
        # Shoulder Lift: Keep it lifted (-pi to 0 is back/up range usually)
        # Default starts at -2.0. Limit to -3.14 to -1.0
        self.target_joint_pos[1] = np.clip(self.target_joint_pos[1], -3.14, -1.0)
        # Elbow: Keep it bent (0 to pi). Default 2.0. Limit 0.5 to 2.8
        self.target_joint_pos[2] = np.clip(self.target_joint_pos[2], 0.5, 2.8)
        # Wrist 1: Pitch (-pi to 0). Default -1.5. Limit -2.5 to -0.5
        self.target_joint_pos[3] = np.clip(self.target_joint_pos[3], -2.5, -0.5)
        # Wrist 2: Roll. Limit lightly +/- 1.57
        self.target_joint_pos[4] = np.clip(self.target_joint_pos[4], -1.57, 1.57)
        # Wrist 3: Yaw. Limit +/- 3.14
        self.target_joint_pos[5] = np.clip(self.target_joint_pos[5], -3.14, 3.14)
        
        self.data.ctrl[:6] = self.target_joint_pos
        
        # Gripper Control - Continuous Mode (Symmetric)
        # 夹爪控制：连续模式，双指对称
        # Action [-1, 1] Mapped to Gripper Range [-0.01, 0.04]
        # -1 -> Close (-0.01)
        #  1 -> Open  (0.04)
        gripper_target = (action[6] + 1.0) * 0.025 - 0.01
        
        self.data.ctrl[6] = gripper_target
        self.data.ctrl[7] = gripper_target

        # Frame Skip: 运行多次物理模拟步，给动作执行的时间 / Run multiple physics steps
        # MuJoCo timestep is usually 0.002s. 
        # range(20) -> 20 * 0.002 = 0.04s per control step (25Hz control frequency)
        for _ in range(20):
            mujoco.mj_step(self.model, self.data)
        
        # --- REWARD CALCULATION ---
        obj_pos = self.data.xpos[self.obj_body_id]
        ee_pos = self.data.site_xpos[self.ee_id]
        
        dist_ee_obj = np.linalg.norm(obj_pos - ee_pos)
        dist_obj_target = np.linalg.norm(obj_pos - self.target_pos)
        
        touching, table_collision = self._check_contact()

        reward = 0
        
        # 0. 安全惩罚 (Safety Penalty)
        if table_collision:
            if not touching:
                reward -= 1.0 # 没抓到物体还撞桌子，给予较重惩罚
            else:
                reward -= 0.1 # 抓取时难免蹭到桌子，轻微惩罚
        
        # 1. 接近奖励 (Distance Reward)
        dist_reward = 1.0 - np.tanh(5.0 * dist_ee_obj)
        reward += 5.0 * dist_reward 
        
        # 2. 姿态奖励 (Orientation Reward)
        # ...existing code...
        site_mat = self.data.site_xmat[self.ee_id].reshape(3, 3)
        y_axis = site_mat[:, 1]
        dot_product = -y_axis[2] # 目标是[0,0,-1]
        reward += np.clip(dot_product, 0, 1) * 2.0  

        # 3. 接触与抓取奖励 (Grasp Incentive)
        gripper_closed_action = action[6] < 0 
        
        if touching:
            reward += 1 # 稍微碰到就有糖吃，鼓励保持接触
            if gripper_closed_action:
                reward += 3.0 
        else:
            # 没摸到时，如果距离很近(<5cm)，鼓励张开夹爪
            if dist_ee_obj < 0.05 and not gripper_closed_action:
                reward += 0.25

        # 4. 抓取成功与提升 (Lift Stage)
        # 物体离地判定：z > 0.45 (桌面约0.4)
        if obj_pos[2] > 0.45:
            # reward += 0.25 
            
            # 搬运奖励：只有离地后才开启
            reward_transport = 1.0 - np.tanh(5.0 * dist_obj_target)
            reward += 5.0 * reward_transport
        
        # 5. 成功条件 / Success Condition
        terminated = False

        # 只要距离满足，即视为到达目标
        if dist_obj_target < 0.05:
            reward += 100.0 # 翻倍大奖(50->100)，确保完成任务是最优策略
            self.on_goal_count += 1
            if self.on_goal_count > 10: # 保持10步以证明稳定性
                terminated = True
        else:
            self.on_goal_count = 0

        # 6. 约束与惩罚 (Constraints & Penalties)
        # 动作惩罚 (Action Penalty)：限制过大的动作幅度和抖动
        reward -= np.sum(np.square(action)) * 0.02

        # 惩罚抬得太高 / Penalty for lifting too high
        # 目标高度0.45，桌面0.4。如果抬到0.65以上，通常是无效动作，扣分以限制动作幅度
        if obj_pos[2] > 0.65:
            reward -= (obj_pos[2] - 0.65) * 10.0
            
        # 6. 失败条件（物体掉落） / Failure Condition
        # 桌子高度为0.4。如果物体掉落到0.3以下，说明在地上或正在掉落。
        if obj_pos[2] < 0.3:
            reward -= 10.0 # 掉落惩罚
            terminated = True # 立即结束回合
            
        truncated = False
        if self.iteration >= 200:
            truncated = True
            
        self.iteration += 1
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        self.viewer.sync()

class RunningMeanStd:
    def __init__(self, shape):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = 1e-4

    def update(self, x):
        # 确保输入是 (Batch, Dim) 形状
        if x.ndim == 1:
            x = x[np.newaxis, :]
            
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

class NormalizationWrapper(gym.Wrapper):
    def __init__(self, env, running_mean_std=None):
        super().__init__(env)
        if running_mean_std is None:
            self.rms = RunningMeanStd(env.observation_space.shape)
        else:
            self.rms = running_mean_std

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.rms.update(obs)
        return self.normalize(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.rms.update(obs)
        return self.normalize(obs), info

    def normalize(self, obs):
        return (obs - self.rms.mean) / np.sqrt(self.rms.var + 1e-8)

    def save_stats(self, path):
        np.savez(path, mean=self.rms.mean, var=self.rms.var, count=self.rms.count)

    def load_stats(self, path):
        data = np.load(path)
        self.rms.mean = data['mean']
        self.rms.var = data['var']
        self.rms.count = data['count']
