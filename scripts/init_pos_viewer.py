# # Copyright (c) 2025, Master Jia
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom robot to an Isaac Lab environment."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

stiffness = {
    "hip_pitch": 400.0,
    "hip_roll": 200.0,
    "hip_yaw": 200.0,
    "knee": 500.0,
    "ankle_pitch": 5,
    "ankle_roll": 0.1,
    "waist": 150,
    "head": 1,
    "arm_pitch_higher": 100,
    "arm_roll": 200,
    "arm_yaw": 5,
    "arm_pitch_lower": 30,
}
damping = {
    "hip_pitch": 50,
    "hip_roll": 20,
    "hip_yaw": 25,
    "knee": 20,
    "ankle_pitch": 3,
    "ankle_roll": 0.03,
    "waist": 10,
    "head": 0.5,
    "arm_pitch_higher": 20,
    "arm_roll": 20,
    "arm_yaw": 2,
    "arm_pitch_lower": 1,
}

V3ALLBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/jf/projects/JF_gym/resource/v3all/urdf/v3alln2/v3alln2.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=[0, 0, 1.0],
        joint_pos={
            # 左腿 (6)
            "left_hip_pitch_joint": -0.2,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": -0.5,
            "left_ankle_pitch_joint": 0.2,
            "left_ankle_roll_joint": 0.0,
            # 右腿 (6)
            "right_hip_pitch_joint": 0.2,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.5,
            "right_ankle_pitch_joint": -0.2,
            "right_ankle_roll_joint": 0.0,
            # 躯干 (2)
            "waist_joint": 0.0,
            "head_joint": 0.0,
            # 左臂 (4)
            "left_arm_pitch_higher_joint": 0.0,
            "left_arm_roll_joint": 0.0,
            "left_arm_yaw_joint": 0.0,
            "left_arm_pitch_lower_joint": 0.0,
            # 右臂 (4)
            "right_arm_pitch_higher_joint": 0.0,
            "right_arm_roll_joint": 0.0,
            "right_arm_yaw_joint": 0.0,
            "right_arm_pitch_lower_joint": 0.0,
        },
    ),
    actuators={
        # === 下肢：髋关节 ===
        "hip_pitch_act": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_hip_pitch_joint",
                "right_hip_pitch_joint",
            ],
            stiffness=stiffness["hip_pitch"],
            damping=damping["hip_pitch"],
            effort_limit_sim=200.0,
            velocity_limit_sim=50.0,
        ),
        "hip_roll_act": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_hip_roll_joint",
                "right_hip_roll_joint",
            ],
            stiffness=stiffness["hip_roll"],
            damping=damping["hip_roll"],
            effort_limit_sim=200.0,
            velocity_limit_sim=50.0,
        ),
        "hip_yaw_act": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_hip_yaw_joint",
                "right_hip_yaw_joint",
            ],
            stiffness=stiffness["hip_yaw"],
            damping=damping["hip_yaw"],
            effort_limit_sim=200.0,
            velocity_limit_sim=50.0,
        ),
        # === 下肢：膝、踝 ===
        "knee_act": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_knee_joint",
                "right_knee_joint",
            ],
            stiffness=stiffness["knee"],
            damping=damping["knee"],
            effort_limit_sim=250.0,
            velocity_limit_sim=60.0,
        ),
        "ankle_pitch_act": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_ankle_pitch_joint",
                "right_ankle_pitch_joint",
            ],
            stiffness=stiffness["ankle_pitch"],
            damping=damping["ankle_pitch"],
            effort_limit_sim=150.0,
            velocity_limit_sim=40.0,
        ),
        "ankle_roll_act": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_ankle_roll_joint",
                "right_ankle_roll_joint",
            ],
            stiffness=stiffness["ankle_roll"],
            damping=damping["ankle_roll"],
            effort_limit_sim=150.0,
            velocity_limit_sim=40.0,
        ),
        # === 躯干、头部 ===
        "waist_act": ImplicitActuatorCfg(
            joint_names_expr=[
                "waist_joint",
            ],
            stiffness=stiffness["waist"],
            damping=damping["waist"],
            effort_limit_sim=100.0,
            velocity_limit_sim=30.0,
        ),
        "head_act": ImplicitActuatorCfg(
            joint_names_expr=[
                "head_joint",
            ],
            stiffness=stiffness["head"],
            damping=damping["head"],
            effort_limit_sim=50.0,
            velocity_limit_sim=20.0,
        ),
        # === 上肢：大臂 pitch / roll / yaw ===
        "arm_pitch_higher_act": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_arm_pitch_higher_joint",
                "right_arm_pitch_higher_joint",
            ],
            stiffness=stiffness["arm_pitch_higher"],
            damping=damping["arm_pitch_higher"],
            effort_limit_sim=120.0,
            velocity_limit_sim=40.0,
        ),
        "arm_roll_act": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_arm_roll_joint",
                "right_arm_roll_joint",
            ],
            stiffness=stiffness["arm_roll"],
            damping=damping["arm_roll"],
            effort_limit_sim=120.0,
            velocity_limit_sim=40.0,
        ),
        "arm_yaw_act": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_arm_yaw_joint",
                "right_arm_yaw_joint",
            ],
            stiffness=stiffness["arm_yaw"],
            damping=damping["arm_yaw"],
            effort_limit_sim=80.0,
            velocity_limit_sim=30.0,
        ),
        # === 上肢：小臂 pitch ===
        "arm_pitch_lower_act": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_arm_pitch_lower_joint",
                "right_arm_pitch_lower_joint",
            ],
            stiffness=stiffness["arm_pitch_lower"],
            damping=damping["arm_pitch_lower"],
            effort_limit_sim=80.0,
            velocity_limit_sim=30.0,
        ),
    },
)


class NewRobotsSceneCfg(InteractiveSceneCfg):
    """Designs the scene."""

    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    V3allbot = V3ALLBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/V3allbot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0

    robot = scene["V3allbot"]

    # --- 准备初始状态数据 ---
    # 既然你已经在 Config 里定义了 init_state，我们直接取出来用
    # 注意：robot.data.default_root_state 和 default_joint_pos
    # 读取的就是你在 ArticulationCfg 里写的 init_state

    # 1. 根节点位置（加上环境偏移）
    default_root_state = robot.data.default_root_state.clone()
    default_root_state[:, :3] += scene.env_origins

    # 2. 关节角度
    default_joint_pos = robot.data.default_joint_pos.clone()
    # 3. 关节速度 (设为0，保持静止)
    default_joint_vel = torch.zeros_like(robot.data.default_joint_vel)

    print("[INFO]: Freezing robot at init_state for inspection...")

    while simulation_app.is_running():
        # === 核心修改点 ===
        # 在每一帧，都强制把机器人的状态写回仿真器
        # 这相当于每一帧都瞬移回初始姿态，视觉上就是静止不动的

        # 1. 锁定根节点位置 (如果你 config 里 fix_root_link=True，这一步其实是可选的，但加上更保险)
        robot.write_root_pose_to_sim(default_root_state[:, :7])
        robot.write_root_velocity_to_sim(default_root_state[:, 7:])

        # 2. 锁定所有关节角度
        robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)

        # 3. 必须调用 write_data_to_sim 才能生效
        scene.write_data_to_sim()

        # 4. 步进物理 (虽然步进了，但下一帧开头我们又把它拽回去了)
        sim.step()

        # 5. 更新场景
        scene.update(sim_dt)
        sim_time += sim_dt


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    # Design scene
    scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
