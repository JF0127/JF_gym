# # Copyright (c) 2025, Master Jia
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.sensors.contact_sensor.contact_sensor import ContactSensor


def feet_air_time(
    env,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """
    奖励脚部在空中的时间。
    只有当机器人被命令移动时（cmd_vel > 0.1），且脚落地那一瞬间，
    根据它之前在空中停留的时间给予奖励。
    """
    # 1. 获取接触传感器数据
    # sensor_cfg.name 必须对应你在 Scene 中定义的 ContactSensor 名字
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # 获取接触力 (Env, Body, 3) -> 取模 -> (Env, Body)
    # 我们只需要知道有没有接触，所以判断力是否大于 1.0 牛顿
    net_contact_forces = contact_sensor.data.net_forces_w_history[:, 0, :, :]
    # 注意：我们要根据 body_ids 筛选出只是“脚”的部分
    # contact_sensor 可能包含全身接触，这里只取配置中指定的索引
    foot_contact = torch.norm(net_contact_forces[:, sensor_cfg.body_ids, :], dim=-1) > 1.0

    # 2. 初始化/获取状态变量 (Monkey Patching)
    # 因为函数是无状态的，我们把计时器 tensor 绑在 env 对象上
    if not hasattr(env, "feet_air_time_state"):
        # 记录每只脚当前在空中停留的时间
        env.feet_air_time_state = torch.zeros_like(foot_contact, dtype=torch.float, device=env.device)
        # 记录上一帧的接触状态，用于检测“落地瞬间”
        env.last_feet_contact = torch.zeros_like(foot_contact, dtype=torch.bool, device=env.device)

    # 3. 更新计时器
    # 如果接触了 (True)，计时器清零；如果没接触 (False)，计时器 + dt
    env.feet_air_time_state += env.step_dt
    env.feet_air_time_state[foot_contact] = 0.0

    # 4. 计算奖励 (只在落地瞬间给予奖励)
    # 落地瞬间 = 当前接触 True AND 上一帧接触 False
    first_contact = foot_contact & ~env.last_feet_contact

    # 奖励值 = (刚才的滞空时间 - 阈值)
    # 只有当滞空时间超过阈值才给正分，否则给 0 (clip)
    # 注意：这里我们用上一帧的 timer 状态，因为这一帧 timer 已经被清零了，
    # 但由于逻辑顺序，我们需要一个临时变量或者稍微调整逻辑。
    # 更简单的做法：直接奖励当前帧还“存活”的 air_time，但这会导致每一帧都给分。
    # 标准做法是：Update Timer -> Check Contact.

    # 修正逻辑：
    # 为了简单且有效，通常我们在接触发生时，查看`last_air_time`。
    # 这里我们采用一种常用的近似方法：奖励 = (AirTime - Threshold).clip(min=0)
    # 但只在 first_contact 为 True 的位置生效。

    # 为了获取清零前的 AirTime，我们在清零前其实应该存一下。
    # 但由于 Python 代码执行顺序，我们这里稍微简化：
    # 使用 env.feet_air_time_state 的值（在清零前）做计算比较复杂。
    # 替代方案：直接给所有“在空中的脚”奖励，但这会导致机器人抬腿不放。

    # 只有在这一帧刚接触地面时，计算奖励
    # 我们利用 first_contact 掩码
    # 此时 env.feet_air_time_state 已经被置 0 了吗？上面代码置0了。
    # 所以我们需要改变一下顺序：

    # A. 增加时间
    env.feet_air_time_state += env.step_dt

    # B. 记录即将发生接触的那些脚的时间
    air_time_at_impact = env.feet_air_time_state.clone()

    # C. 对接触地面的脚清零
    env.feet_air_time_state[foot_contact] = 0.0

    # D. 计算奖励：只针对刚落地的脚
    rew = (air_time_at_impact - threshold).clamp(min=0.0)
    rew = rew * first_contact.float()  # 只保留刚落地的那一刻的奖励

    # 5. 过滤指令
    # 如果 Command 指令显示机器人应该停止 (速度很小)，就不应该奖励抬腿
    # 获取指令: (Env, 3) -> [vx, vy, w]
    commands = env.command_manager.get_command(command_name)
    # 计算水平速度命令的大小
    cmd_vel_norm = torch.norm(commands[:, :2], dim=-1)  # x, y 速度

    # 如果命令速度 < 0.1 m/s，奖励置 0 (防止原地踏步赚分)
    rew[cmd_vel_norm < 0.1] = 0.0

    # 6. 更新上一帧状态
    env.last_feet_contact = foot_contact.clone()

    # 7. 求和返回 (对所有脚的奖励求和)
    return torch.sum(rew, dim=-1)
