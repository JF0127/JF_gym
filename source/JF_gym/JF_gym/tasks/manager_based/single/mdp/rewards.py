# # Copyright (c) 2025, Master Jia
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations


def track_single_joint_position(env, joint_name: str, target: float):
    """单关节位置跟踪奖励： - (q - target)^2

    Args:
        env: ManagerBasedRLEnv 实例
        term_cfg: RewardTermCfg（不用的话可以不管）
        joint_name: 关节名
        target: 目标角度（弧度）
    """
    # 取出机器人 articulation
    robot = env.scene["robot"]  # 对应 SceneCfg 里 robot 的名字

    # 所有关节位置: [num_envs, num_dofs]
    joint_pos = robot.data.joint_pos

    # 找到这个关节的下标
    # 这里假设你有 dof 名字列表，如果名字不一样，你可以改成自己项目现有的方式
    dof_names = robot.data.joint_names  # 有的版本叫 dof_names，需要你对照一下
    if isinstance(dof_names, list):
        idx = dof_names.index(joint_name)
    else:
        # 如果 joint_names 是numpy数组或别的结构，你可能需要自己改这里
        idx = list(dof_names).index(joint_name)

    q = joint_pos[:, idx]  # [num_envs]
    error = q - target  # [num_envs]
    reward = -error * error  # [num_envs]

    return reward
