# UR5E 机械臂强化学习抓取与放置项目 (TD3 + MuJoCo)

本项目基于 **TD3 (Twin Delayed Deep Deterministic Policy Gradient)** 算法，使用 Gymnasium 和 MuJoCo 物理引擎，训练 UR5E 机械臂完成 **抓取 (Pick)** 和 **放置 (Place)** 任务。

---

## 📅 项目介绍

### 任务目标
控制 UR5E 机械臂的 6 个关节自由度以及末端夹爪，从桌面上抓取一个随机位置的红色方块，并将其搬运到指定的蓝色目标区域。环境中可能存在多个随机颜色的障碍物，增加了任务的难度。

### 核心算法: TD3
本项目采用 TD3 算法，它是 DDPG (Deep Deterministic Policy Gradient) 的改进版本，主要解决了 DDPG 中的 Q 值高估问题，并在连续动作空间任务中表现出色。
TD3 的三个关键改进：
1.  **双 Critic 网络 (Clipped Double Q-Learning)**: 使用两个 Critics ($Q_1, Q_2$) 并取最小值计算目标 Q 值，缓解过估计。
2.  **延迟 Actor 更新 (Delayed Policy Updates)**: Critic 更新频率高于 Actor，确保 Actor 在更准确的价值评估基础上进行优化。
3.  **目标策略平滑 (Target Policy Smoothing)**: 在目标动作中加入截断的高斯噪声，防止 Policy 对 Q 值函数的误差过拟合。

---

## 🧠 网络架构

本项目采用全连接神经网络 (MLP) 构建 Actor 和 Critic。

### 1. Actor 网络 (策略网络)
*   **输入**: 状态向量 (State, 25维)
*   **输出**: 动作向量 (Action, 7维, $\in [-1, 1]$)
*   **结构**:
| 层级 | 类型 | 节点数 | 激活函数 | 备注 |
| :--- | :--- | :--- | :--- | :--- |
| Layer 1 | Linear | $25 \to 512$ | ReLU | 状态特征提取 |
| Layer 2 | Linear | $512 \to 512$ | ReLU | 深层特征提取 |
| Layer 3 | Linear | $512 \to 512$ | ReLU | 新增层，增强拟合能力 |
| Output | Linear | $512 \to 7$ | Tanh | 缩放至 [-1, 1] |

### 2. Critic 网络 (价值网络)
本项目使用了两个结构相同的 Critic 网络 ($Q_1$ 和 $Q_2$)。
*   **输入**: 状态向量 (25维) + 动作向量 (7维) = 32维
*   **输出**: Q 值 (1维标量)
*   **结构**:
| 层级 | 类型 | 节点数 | 激活函数 | 备注 |
| :--- | :--- | :--- | :--- | :--- |
| Layer 1 | Linear | $32 \to 512$ | ReLU | State & Action 融合 |
| Layer 2 | Linear | $512 \to 512$ | ReLU | 特征提取 |
| Layer 3 | Linear | $512 \to 512$ | ReLU | 特征提取 |
| Output | Linear | $512 \to 1$ | Linear | 输出预测的 Q 值 |

---

## 📊 状态空间 (Observation Space)

状态空间为一个 25 维的向量，包含以下信息：

1.  **机械臂关节位置 (6维)**: `data.qpos[:6]` (弧度)
2.  **机械臂关节速度 (6维)**: `data.qvel[:6]` (弧度/秒)
3.  **夹爪宽度 (1维)**: 左右指位置之和，反映夹爪开合状态。
4.  **物体绝对位置 (3维)**: 红色方块在世界坐标系下的 $(x, y, z)$ 坐标。
5.  **相对位置 (EE -> Obj) (3维)**: 末端执行器到物体的向量差 $(x_{obj}-x_{ee}, ...)$，这是最重要的视觉引导信息。
6.  **相对位置 (Obj -> Target) (3维)**: 物体到目标点的向量差 $(x_{tgt}-x_{obj}, ...)$。
7.  **末端执行器姿态 (3维)**: 末端执行器坐标系的 Y 轴向量 (从指尖指向下方)，用于辅助调整抓取角度。

---

## 📐 奖励函数设计 (Reward Function)

奖励函数经过精心设计，采用 **稠密奖励 (Dense Reward)** 引导策略逐步学习。

总奖励 $R = R_{approach} + R_{orient} + R_{grasp} + R_{transport} + R_{success} + R_{penalty}$

#### 1. 接近奖励 (Distance Reward)
鼓励机械臂末端靠近物体，使用 Double Tanh 缩放：
$$ R_{approach} = 5.0 \times (1.0 - \tanh(5.0 \times d_{ee\_obj})) $$
其中 $d_{ee\_obj}$ 是末端与物体的欧几里得距离。

#### 2. 姿态奖励 (Orientation Reward)
鼓励抓取时爪子垂直向下 (Open downwards)：
$$ R_{orient} = 2.0 \times \text{clip}(\vec{v}_{finger} \cdot \vec{v}_{down}, 0, 1) $$
其中 $\vec{v}_{down} = [0, 0, -1]$。

#### 3. 抓取与接触奖励 (Contact & Grasp)
*   **接触奖励**: 只要碰到物体，奖励 $+1.0$。
*   **有效抓取**: 碰到物体且如果处于闭合夹爪状态，奖励 $+3.0$。
*   **预抓取姿态**: 未接触但距离很近 ($<5$cm)且张开夹爪，奖励 $+0.25$。

#### 4. 搬运奖励 (Transport Reward)
只有当物体被抬离桌面 ($z > 0.45$) 时才激活此奖励，引导物体向目标点移动：
$$ R_{transport} = 5.0 \times (1.0 - \tanh(5.0 \times d_{obj\_target})) $$

#### 5. 成功奖励 (Success Reward)
当物体距离目标点小于 $5$cm 时触发：
*   **大奖**: $R_{success} = +100.0$
*   **稳定标识**: 连续保持 $10$ 步满足条件才算成功结束 (Terminated)。

#### 6. 惩罚 (Penalties)
*   **安全惩罚**: 撞击桌子且未抓到物体 $R_{safety} = -1.0$。
*   **掉落惩罚**: 物体掉出桌面 (失败) $R_{drop} = -10.0$。
*   **动作平滑**: 惩罚过大的动作输出 $R_{ctrl} = -0.02 \times \|a\|^2$。

---

## 📉 损失函数与优化 (Loss & Optimization)

### Critic Loss Function
我们选择 **`SmoothL1Loss` (Huber Loss)** 而不是传统的 `MSELoss`。
$$ L_{critic} = \frac{1}{N} \sum \text{SmoothL1}(Q_{pred}, Q_{target}) $$
**选择理由**:
*   在训练初期或由于探索导致的异常样本（Target Q 偏差极大）出现时，MSE 会产生巨大的梯度，导致网络权重震荡甚至发散。
*   SmoothL1 在误差较大时表现为 L1 范数（线性，梯度恒定），限制了梯度的爆炸，使 Critic 的训练更加鲁棒和稳定。

### Actor Loss Function
使用确定性策略梯度的目标函数：
$$ L_{actor} = -\frac{1}{N} \sum Q_1(s, \pi(s)) $$
即最大化 Critic $Q_1$ 对当前 Actor 输出动作的评分。

---

## 🛠️ 超参数 (Hyperparameters)

*   **Actor Learning Rate**: `1e-4`
*   **Critic Learning Rate**: `1e-3`
*   **Gamma (折扣因子)**: `0.98` (关注较长远的收益)
*   **Tau (软更新系数)**: `0.005`
*   **Batch Size**: `512`
*   **Replay Buffer Size**: `1,000,000`
*   **Exploration Noise**: `0.1` (Gaussian)

## 🚀 快速开始

### 训练 (Training)
```bash
python train.py
```
这会启动多环境并行采样训练，并利用专家演示数据 (Demonstrations) 进行预训练（似乎可以删除）。

### 测试 (Testing)
```bash
python test.py
```
这会加载训练好的模型 (`TD3_UR5E_Actor.pth`) 并运行一个带有 **若干个随机干扰障碍物** 的测试环境。
