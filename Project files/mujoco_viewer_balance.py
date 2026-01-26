#!/usr/bin/env python3
"""
GO2-W STANDING UP / LAYING DOWN / BALANCING - MUJOCO NATIVE VIEWER
"""

import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path('unitree_mujoco/unitree_robots/go2w/scene.xml')
data = mujoco.MjData(model)

mujoco.mj_resetData(model, data)

# Initial position
data.qpos[0] = 0.0
data.qpos[1] = 0.0
data.qpos[2] = 0.45

# Upright orientation
data.qpos[3] = 1.0
data.qpos[4] = 0.0
data.qpos[5] = 0.0
data.qpos[6] = 0.0

# Define poses
POSES = {
    'lying': {
        'z': 0.15,
        'front_hip': 0.0,
        'front_thigh': 1.4,
        'front_calf': -2.8,
        'rear_hip': 0.0,
        'rear_thigh': 1.4,
        'rear_calf': -2.8
    },
    'sitting': {
        'z': 0.25,
        'front_hip': 0.0,
        'front_thigh': 0.8,
        'front_calf': -1.6,
        'rear_hip': 0.0,
        'rear_thigh': 1.2,
        'rear_calf': -2.2
    },
    'standing': {
        'z': 0.45,
        'front_hip': 0.0,
        'front_thigh': 0.8,
        'front_calf': -1.6,
        'rear_hip': 0.0,
        'rear_thigh': 0.8,
        'rear_calf': -1.6
    },
    'pre_balance': {
        'z': 0.28,
        'front_hip': 0.0,
        'front_thigh': 1.6,   # Front legs crouched low
        'front_calf': -2.8,   # Front very compressed
        'rear_hip': 0.9,      # Hip bent forward 
        'rear_thigh': 1.5,    # Thigh bent to tuck under
        'rear_calf': -2.6     # Wheels close to front
    },
    'mid_balance': {
        'z': 0.35,
        'front_hip': 0.0,
        'front_thigh': 1.0,   # Front legs pull back slightly
        'front_calf': -1.8,   # Still bent but gaining support
        'rear_hip': 0.7,      # Keep rear tucked close
        'rear_thigh': 1.2,    # Rear lifting off ground
        'rear_calf': -2.0     # Rear wheels coming up
    },
    'balancing': {
        'z': 0.50,
        'front_hip': 0.0,
        'front_thigh': 0.6,   # Front legs fully extended
        'front_calf': -1.2,   # Strong support position
        'rear_hip': 0.6,      # Rear pulled close to body
        'rear_thigh': 0.8,    # Rear lifted high
        'rear_calf': -1.4     # Rear wheels well off ground
    }
}

# Actuator mappings
ACT_TO_QPOS = {
    0: 11,  1: 12,  2: 13,   # FR
    3: 7,   4: 8,   5: 9,    # FL
    6: 19,  7: 20,  8: 21,   # RR
    9: 15, 10: 16, 11: 17,   # RL
}

ACT_TO_QVEL = {
    0: 10,  1: 11,  2: 12,   # FR
    3: 6,   4: 7,   5: 8,    # FL
    6: 18,  7: 19,  8: 20,   # RR
    9: 14, 10: 15, 11: 16,   # RL
}

# Map actuators to front/rear
LEG_TYPE = {
    0: 'front',  1: 'front',  2: 'front',   # FR
    3: 'front',  4: 'front',  5: 'front',   # FL
    6: 'rear',   7: 'rear',   8: 'rear',    # RR
    9: 'rear',  10: 'rear',  11: 'rear',    # RL
}

ACT_TO_JOINT = {
    0: 'hip',   1: 'thigh',  2: 'calf',   # FR
    3: 'hip',   4: 'thigh',  5: 'calf',   # FL
    6: 'hip',   7: 'thigh',  8: 'calf',   # RR
    9: 'hip',  10: 'thigh', 11: 'calf',   # RL
}

# Higher gains for balancing stability
KP = 800.0
KD = 30.0

# Balance feedback gains
BALANCE_KP = 80.0  # Proportional gain for pitch correction
BALANCE_KD = 20.0  # Derivative gain for pitch rate damping

def interpolate_pose(pose_start, pose_end, t):
    """Interpolate between two poses"""
    t = np.clip(t, 0.0, 1.0)
    pose = {}
    for key in pose_start:
        pose[key] = pose_start[key] + t * (pose_end[key] - pose_start[key])
    return pose

def set_initial_pose(pose_name):
    """Set robot to initial pose"""
    pose = POSES[pose_name]
    data.qpos[2] = pose['z']
    
    # Front legs (FL, FR)
    for leg_start in [7, 11]:
        data.qpos[leg_start]     = pose['front_hip']
        data.qpos[leg_start + 1] = pose['front_thigh']
        data.qpos[leg_start + 2] = pose['front_calf']
        data.qpos[leg_start + 3] = 0.0
    
    # Rear legs (RL, RR)
    for leg_start in [15, 19]:
        data.qpos[leg_start]     = pose['rear_hip']
        data.qpos[leg_start + 1] = pose['rear_thigh']
        data.qpos[leg_start + 2] = pose['rear_calf']
        data.qpos[leg_start + 3] = 0.0
    
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

# Start in standing pose
set_initial_pose('standing')

# Animation state
current_pose = 'standing'
target_pose = 'standing'
pose_sequence = []  # Queue of poses to transition through
transition_start_time = None
TRANSITION_DURATION = 1.2  # Duration for each stage

def controller(model, data):
    """PD controller with balance feedback"""
    global current_pose, target_pose, transition_start_time, pose_sequence
    
    # Determine current target positions
    if transition_start_time is not None:
        elapsed = data.time - transition_start_time
        t = elapsed / TRANSITION_DURATION
        
        if t >= 1.0:
            current_pose = target_pose
            
            # Check if there are more poses in the sequence
            if len(pose_sequence) > 0:
                target_pose = pose_sequence.pop(0)
                transition_start_time = data.time
                stage_num = 4 - len(pose_sequence)  # For logging
                print(f"Stage {stage_num}: Transitioning to {target_pose}... (time={data.time:.2f})")
            else:
                transition_start_time = None
                print(f"Transition complete: now {current_pose}")
            t = 1.0
        
        pose = interpolate_pose(POSES[current_pose], POSES[target_pose], t)
    else:
        pose = POSES[current_pose]
    
    # Balance feedback when in balancing pose
    pitch_correction = 0.0
    if current_pose == 'balancing' or (target_pose == 'balancing' and transition_start_time is not None):
        # Get pitch angle (rotation around Y-axis)
        # Quaternion is [w, x, y, z] at indices [3, 4, 5, 6]
        qw, qx, qy, qz = data.qpos[3:7]
        
        # Convert quaternion to pitch (rotation around Y-axis)
        pitch = np.arcsin(2.0 * (qw * qy - qz * qx))
        
        # Get pitch rate (angular velocity around Y-axis)
        pitch_rate = data.qvel[4]  # Y-axis angular velocity
        
        # Target pitch: moderate backward lean for front-wheel balance
        target_pitch = -0.2  # ~11.5 degrees backward (negative = rear up)
        
        # Calculate correction (positive = lean back, negative = lean forward)
        pitch_error = pitch - target_pitch
        pitch_correction = BALANCE_KP * pitch_error + BALANCE_KD * pitch_rate
    
    # Build targets with balance feedback
    for act_idx in range(12):
        leg_type = LEG_TYPE[act_idx]
        joint_type = ACT_TO_JOINT[act_idx]
        
        # Get base target from pose
        if leg_type == 'front':
            target = pose[f'front_{joint_type}']
        else:
            target = pose[f'rear_{joint_type}']
        
        # Apply balance correction to front legs' thigh joints
        if current_pose == 'balancing' and leg_type == 'front' and joint_type == 'thigh':
            target -= pitch_correction * 0.03  # Increased correction strength
        
        # PD control
        qpos_idx = ACT_TO_QPOS[act_idx]
        qvel_idx = ACT_TO_QVEL[act_idx]
        
        pos = data.qpos[qpos_idx]
        vel = data.qvel[qvel_idx]
        
        torque = KP * (target - pos) - KD * vel
        torque = np.clip(torque, -23.0, 23.0)
        
        data.ctrl[act_idx] = torque
    
    # Wheels off
    for i in range(12, 16):
        data.ctrl[i] = 0.0

def key_callback(keycode):
    """Handle keyboard input"""
    global target_pose, transition_start_time, current_pose, pose_sequence
    
    if transition_start_time is not None:
        return  # Already transitioning
    
    # MuJoCo uses ASCII codes
    # U = 85, I = 73, L = 76, B = 66, P = 80
    # u = 117, i = 105, l = 108, b = 98, p = 112
    
    if keycode in [85, 117]:  # U or u
        if current_pose != 'standing':
            target_pose = 'standing'
            pose_sequence = []
            transition_start_time = data.time
            print(f"Command: Standing up... (time={data.time:.2f})")
    elif keycode in [73, 105]:  # I or i
        if current_pose != 'sitting':
            target_pose = 'sitting'
            pose_sequence = []
            transition_start_time = data.time
            print(f"Command: Sitting down... (time={data.time:.2f})")
    elif keycode in [76, 108]:  # L or l
        if current_pose != 'lying':
            target_pose = 'lying'
            pose_sequence = []
            transition_start_time = data.time
            print(f"Command: Laying down... (time={data.time:.2f})")
    elif keycode in [80, 112]:  # P or p
        if current_pose != 'pre_balance':
            target_pose = 'pre_balance'
            pose_sequence = []
            transition_start_time = data.time
            print(f"Command: Pre-balance crouch... (time={data.time:.2f})")
    elif keycode in [66, 98]:  # B or b
        if current_pose != 'balancing':
            # Three-stage transition
            target_pose = 'pre_balance'
            pose_sequence = ['mid_balance', 'balancing']
            transition_start_time = data.time
            print(f"Command: Balancing (Stage 1: Crouch with wheels close)... (time={data.time:.2f})")

print("=" * 60)
print("KEYBOARD CONTROL")
print("=" * 60)
print("Press keys while the viewer window has focus:")
print("  U - Stand up")
print("  I - Sit down")
print("  L - Lay down")
print("  P - Pre-balance crouch (front low, rear wheels close)")
print("  B - Balance on front wheels (3-stage sequence)")
print("\nPress ESC or close window to exit")
print("=" * 60)

# Launch viewer with controller
with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    while viewer.is_running():
        # Run controller
        controller(model, data)
        
        # Step simulation
        mujoco.mj_step(model, data)
        
        # Sync viewer (updates display)
        viewer.sync()

print("Done!")