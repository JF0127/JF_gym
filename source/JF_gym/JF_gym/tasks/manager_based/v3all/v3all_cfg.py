# # Copyright (c) 2025, Master Jia
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg

# 引入 ContactSensorCfg，因为您的代码依赖 contact_forces
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from . import mdp
from .robots.v3all import V3ALLBOT_CONFIG

##
# Scene definition
##


@configclass
class JfGymSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # -----------------------------------------------------------------
    # 修改开始：不仅是 replace路径，还要开启 contact sensors
    # -----------------------------------------------------------------

    # 1. 复制并修改路径
    v3all_cfg: ArticulationCfg = V3ALLBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 2. [关键] 强制开启接触传感器支持
    # 这会告诉 PhysX 引擎：请计算并缓存这个资产的接触力数据
    v3all_cfg.spawn.activate_contact_sensors = True

    # 3. 赋值给场景实体
    v3all: ArticulationCfg = v3all_cfg

    # -----------------------------------------------------------------
    # 修改结束
    # -----------------------------------------------------------------

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # Contact Sensor
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="v3all",  # 正确
        joint_names=[".*"],
        scale=0.5,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # 为了避免下面写太多重复代码，定义一个针对 v3all 的配置对象
        # 这是一个小技巧，让代码更整洁

        # 1. 任务指令
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        # 2. 姿态感知 [修改：指定 asset_cfg]
        projected_gravity = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("v3all")})

        # 3. 基座线速度 [修改：指定 asset_cfg]
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("v3all")})

        # 4. 基座角速度 [修改：指定 asset_cfg]
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("v3all")})

        # 5. 关节信息 [修改：指定 asset_cfg]
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("v3all")})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("v3all")})

        # 6. 上一步动作 [无需修改，因为它读的是 Env 的 buffer]
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="v3all",  # 正确
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
            heading=(0.0, 0.0),
        ),
    )


@configclass
class EventCfg:
    """Configuration for events."""

    # Reset Base [需要修改 params 里的 key]
    # 注意：reset_root_state_uniform 并没有 asset_cfg 参数，
    # 它依赖 filter (asset_cfg) 传入 SceneEntityCfg
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
            # [关键] 这里需要指定重置谁
            "asset_cfg": SceneEntityCfg("v3all"),
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("v3all", joint_names=[".*"]),
            "position_range": (0.5, 1.5),
            "velocity_range": (-0.1, 0.1),
        },
    )

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("v3all", body_names=".*"),
            "static_friction_range": (0.4, 1.0),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 3.0),
        params={
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
            # [关键] 指定推谁
            "asset_cfg": SceneEntityCfg("v3all"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # 1. 速度追踪 [修改：指定 asset_cfg]
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.5,
        # 注意：这里既要指定 command_name，也要指定 asset_cfg (谁在追踪)
        params={"command_name": "base_velocity", "std": 0.25, "asset_cfg": SceneEntityCfg("v3all")},
    )

    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.8,
        params={"command_name": "base_velocity", "std": 0.25, "asset_cfg": SceneEntityCfg("v3all")},
    )

    # [修改 1] 脚部滞空 (Feet Air Time)
    # 之前的正则 ".*_foot" 也会报错，因为你的机器人里没有 foot，只有 ankle
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.5,
        params={
            # 修改：将 .*_foot 改为 .*_ankle_roll_link (假设 roll link 是末端脚掌)
            # 如果你的机器人 pitch link 也是脚掌的一部分，可以写 .*_ankle_.*
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )

    # [修改 2] 非足部碰撞惩罚 (Undesired Contacts)
    # 正则需要与 TerminationsCfg 里的保持一致 (除了脚以外的所有部位)
    # [防摔倒] 非足部碰撞惩罚 (Undesired Contacts)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            # 1. 传感器配置 (记得用你刚才修正过的正则)
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names="base_link|waist_link|head_link|.*_arm_.*|.*_hip_.*|.*_knee_link"
            ),
            # 2. [新增] 必须加上这个阈值！
            "threshold": 1.0,
        },
    )

    # 3. 惩罚项 [全部修改：指定 asset_cfg]

    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0, params={"asset_cfg": SceneEntityCfg("v3all")})

    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05, params={"asset_cfg": SceneEntityCfg("v3all")})

    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-0.0002, params={"asset_cfg": SceneEntityCfg("v3all")})

    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
        # action_rate 读取的是 Actuator 历史，通常不需要 explicit asset_cfg，
        # 但如果报错，也可以加上 params={"asset_cfg": SceneEntityCfg("v3all")}
    )

    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": SceneEntityCfg("v3all")})

    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": SceneEntityCfg("v3all")})


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # [修改] 摔倒检测：非法接触
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            # 修改这里的正则以匹配 v3all 的骨骼名称
            # 逻辑：除了脚 (ankle) 以外，其他部位碰到地面都算摔倒
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names="base_link|waist_link|head_link|.*_arm_.*|.*_hip_.*|.*_knee_link"
            ),
            "threshold": 1.0,
        },
        time_out=False,
    )

    base_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("v3all")},
        time_out=False,
    )


# JfGymEnvCfg 保持不变 (记得取消掉 PhysX 那段注释)
@configclass
class JfGymEnvCfg(ManagerBasedRLEnvCfg):
    # -------------------------------------------------------------------------
    # 1. 场景设置 (Scene)
    # -------------------------------------------------------------------------
    scene: JfGymSceneCfg = JfGymSceneCfg(num_envs=4096, env_spacing=4.0)

    # -------------------------------------------------------------------------
    # 2. MDP 组件 (Observations, Actions, Commands, Events)
    # -------------------------------------------------------------------------
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()

    # [新增] 必须加上这个！否则 Reward 和 Observation 里的 command 没法用
    commands: CommandsCfg = CommandsCfg()

    events: EventCfg = EventCfg()

    # -------------------------------------------------------------------------
    # 3. 奖励与终止 (Rewards, Terminations)
    # -------------------------------------------------------------------------
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # -------------------------------------------------------------------------
    # 4. 后处理初始化 (Post Initialization)
    # -------------------------------------------------------------------------
    def __post_init__(self) -> None:
        """Post initialization."""
        # --- 核心仿真参数 ---
        self.decimation = 10  # 抽帧数
        self.sim.dt = 0.001  # 物理仿真步长 (1ms = 1000Hz)

        # 控制频率 = 1 / (dt * decimation) = 1 / 0.01 = 100Hz
        # 100Hz 对于双足机器人是非常理想的控制频率 (Sim-to-Real 标准)

        self.sim.render_interval = self.decimation

        # --- 训练参数调整 ---
        # [修改] 5秒太短了！
        # 机器人刚站稳可能就重置了，学不到长距离行走的稳定性。
        # 推荐：20秒 (2000 steps @ 100Hz) 是双足行走的标准设置。
        self.episode_length_s = 20.0

        # --- 视觉设置 ---
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.viewer.lookat = (0.0, 0.0, 1.0)  # 让相机看向机器人中心
