# # Copyright (c) 2025, Master Jia
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

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
            disable_gravity=False,
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
