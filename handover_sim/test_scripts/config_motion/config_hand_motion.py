import pathlib
from enum import Enum

CONFIG_ROOT = pathlib.Path(__file__).parent.resolve()

class StageConfig(Enum):
    BASIC = 0
    COLORED = 1
    DETAILED = 2
stage_mapping = {
    StageConfig.BASIC: "stage2_mock.usd",
    StageConfig.COLORED: "stage_colored.usd",
    StageConfig.DETAILED: "stage_room.usd",
}
robot_prim_mapping = {
    StageConfig.BASIC: "/World/ur10_camera_holder_robotiq85/robot",
    StageConfig.COLORED: "/World/stage2/ur10_camera_holder_robotiq85/robot",
    StageConfig.DETAILED: "/World/stage2/ur10_camera_holder_robotiq85/robot",
}

class HandConfig(Enum):
    MANO_BASIC = 0
    MANO_COARSE = 1
    MANO_FINE = 2
hand_mapping = {
    HandConfig.MANO_BASIC: "mano_realistic/mano.usd",
    HandConfig.MANO_COARSE: "mano_realistic/mano_coarse.usd",
    HandConfig.MANO_FINE: "mano_realistic/mano_fine.usd",
}

class JointsROS2CommandTopic(Enum):
    MANUAL_JOINTS_CONTROL = 0
    MOCK_HARDWARE = 1
    NO_CONTROL = 2
    # TOPIC_BASED = 3 # For the topic based to work, the joint_states have to be published manually
joints_topic_mapping = {
    JointsROS2CommandTopic.MANUAL_JOINTS_CONTROL: "joint_command",
    JointsROS2CommandTopic.MOCK_HARDWARE: "joint_states",
    # JointsROS2CommandTopic.TOPIC_BASED: "isaac_joint_commands",
    JointsROS2CommandTopic.NO_CONTROL: None,
}

STAGE_CONFIG = StageConfig.BASIC
JOINTS_CONFIG = JointsROS2CommandTopic.MOCK_HARDWARE
HAND_CONFIG = HandConfig.MANO_FINE
JOINTS_COMMAND_TOPIC = joints_topic_mapping[JOINTS_CONFIG]
ISAAC_EXPERIENCE = "isaaclab.python.rendering.kit" # Refer to README.md for other options
SCENE_DIR = CONFIG_ROOT / '../../assets/motions/genh2r'
SCENE_ID = 46
FRAME_INTERVAL = 10  # interval between frames when loading motion data
STAGE_DIR = CONFIG_ROOT / '../../assets/usd/stage'
ROBOT_STAGE_DIR = CONFIG_ROOT / '../../assets/usd/robot'
ROBOT_STAGE_USD_FILE = ROBOT_STAGE_DIR / stage_mapping[STAGE_CONFIG]  # refer to README.md for stage options
ROBOT_PRIM_PATH = robot_prim_mapping[STAGE_CONFIG]
HAND_USD_DIR = CONFIG_ROOT / '../../assets/usd/hand'
HAND_RIGHT_USD_FILE = hand_mapping[HandConfig.MANO_FINE]
HAND_LEFT_USD_FILE = 'mano_realistic_left/mano_left.usd' # TODO: add left hand options
OBJ_USD_DIR = CONFIG_ROOT / '../../assets/usd/objects/ycb_ported'
hand_joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint17', 'joint26', 'joint35',
               'joint44', 'joint8', 'joint18', 'joint27', 'joint36', 'joint45', 'joint9', 'joint19', 'joint28',
               'joint37', 'joint46', 'joint10', 'joint20', 'joint29', 'joint38', 'joint47', 'joint11', 'joint21', 
               'joint30', 'joint39', 'joint48', 'joint12', 'joint22', 'joint31', 'joint40', 'joint49', 'joint13', 
               'joint23', 'joint32', 'joint41', 'joint50', 'joint14', 'joint24', 'joint33', 'joint42', 'joint51', 
               'joint15', 'joint25', 'joint34', 'joint43', 'joint52', 'joint16']
obj_joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
hand_joint_stiffness = {joint_name: 10. for joint_name in hand_joint_names}
hand_joint_damping = None
obj_joint_stiffness = {joint_name: 1.0 for joint_name in obj_joint_names}
obj_joint_damping = None #{joint_name: 0.1 for joint_name in obj_joint_names}
LEFT_MASS = 0.00001
ros2_mock_robot_joint_names_mapping = {'finger_joint': 0,
                               'shoulder_pan_joint': 1,
                               'wrist_3_joint': 2,
                               'wrist_2_joint': 3,
                               'wrist_1_joint': 4,
                               'elbow_joint': 5,
                               'shoulder_lift_joint': 6}
robot_joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint",
                             "wrist_2_joint", "wrist_3_joint", "finger_joint",
                             "right_outer_knuckle_joint", "left_outer_finger_joint", "right_outer_finger_joint",
                             "left_inner_finger_joint", "right_inner_finger_joint",
                             "left_inner_finger_knuckle_joint", "right_inner_finger_knuckle_joint"]
# change poses to:
robot_qpos = {'shoulder_pan_joint': 0.0, 'shoulder_lift_joint': -3.2117, 'elbow_joint': 2.3741, 'wrist_1_joint': 0.979, 
              'wrist_2_joint': 1.5398, 'wrist_3_joint': 3.1816, 'finger_joint': 0.0189, 'right_outer_knuckle_joint': 0.0199, 
              'left_outer_finger_joint': 0.0, 'right_outer_finger_joint': 0.0, 'left_inner_finger_joint': -0.0283,
              'right_inner_finger_joint': 0.0294, 'left_inner_finger_knuckle_joint': -0.0279, 'right_inner_finger_knuckle_joint': -0.029}
robot_joint_stiffness = {joint_name: 50.0 for joint_name in robot_joint_names}
robot_joint_damping = {joint_name: 1.0 for joint_name in robot_joint_names}
