#!/usr/bin/env python3
"""
GO2-W REALISTIC SPIN - MUJOCO NATIVE VIEWER
Spins by paddling legs outward in coordinated pattern
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
        'hip': 0.0,
        'thigh': 1.4,
        'calf': -2.8
    },
    'sitting': {
        'z': 0.25,
        'hip': 0.0,
        'thigh': 1.2,
        'calf': -2.2
    },
    'standing': {
        'z': 0.45,
        'hip': 0.0,
        'thigh': 0.8,
        'calf': -1.6
    },
    'spin_crouch': {
        'z': 0.30,
        'hip': 0.0,
        'thigh': 1.0,
        'calf': -2.0
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

ACT_TO_JOINT = {
    0: 'hip',   1: 'thigh',  2: 'calf',   # FR
    3: 'hip',   4: 'thigh',  5: 'calf',   # FL
    6: 'hip',   7: 'thigh',  8: 'calf',   # RR
    9: 'hip',  10: 'thigh', 11: 'calf',   # RL
}

# Leg groups for coordinated spinning
# FR and RL push right, FL and RR push left
LEG_GROUPS = {
    'group_a': [0, 9],   # FR, RL - push clockwise
    'group_b': [3, 6]    # FL, RR - push counter-clockwise
}

KP = 500.0
KD = 20.0

# Spin parameters
SPIN_CYCLE_TIME = 1.5  # seconds for one complete paddle cycle
SPIN_HIP_AMPLITUDE = 0.4  # radians - how far hips swing out

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
    
    for leg_start in [7, 11, 15, 19]:
        data.qpos[leg_start]     = pose['hip']
        data.qpos[leg_start + 1] = pose['thigh']
        data.qpos[leg_start + 2] = pose['calf']
        data.qpos[leg_start + 3] = 0.0
    
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

# Start in standing pose
set_initial_pose('standing')

# Animation state
current_pose = 'standing'
target_pose = 'standing'
transition_start_time = None
TRANSITION_DURATION = 2.0

# Spin state
is_spinning = False
spin_start_time = 0.0

# Jump state
is_jumping = False
jump_start_time = 0.0
JUMP_CROUCH_TIME = 0.3  # Time to crouch before jump
JUMP_PUSH_TIME = 0.15   # Time to push off ground
JUMP_FLIGHT_TIME = 0.4  # Approximate time in air
JUMP_TOTAL_TIME = JUMP_CROUCH_TIME + JUMP_PUSH_TIME + JUMP_FLIGHT_TIME

def get_spin_hip_offset(leg_actuator_idx, phase):
    """
    Calculate hip offset for spinning motion
    phase: 0 to 1 representing position in cycle
    """
    # Group A (FR, RL) and Group B (FL, RR) move in opposite directions
    if leg_actuator_idx in LEG_GROUPS['group_a']:
        # Push outward (positive hip angle)
        offset = SPIN_HIP_AMPLITUDE * np.sin(2 * np.pi * phase)
    else:  # group_b
        # Push outward opposite direction (negative hip angle)
        offset = -SPIN_HIP_AMPLITUDE * np.sin(2 * np.pi * phase)
    
    return offset

def get_jump_pose(jump_elapsed):
    """
    Calculate pose for jumping motion
    Returns dict with thigh/calf angles
    """
    if jump_elapsed < JUMP_CROUCH_TIME:
        # Phase 1: Crouch down
        t = jump_elapsed / JUMP_CROUCH_TIME
        return {
            'hip': 0.0,
            'thigh': 0.8 + t * 0.6,  # 0.8 -> 1.4 (bend more)
            'calf': -1.6 - t * 1.2   # -1.6 -> -2.8 (bend more)
        }
    elif jump_elapsed < JUMP_CROUCH_TIME + JUMP_PUSH_TIME:
        # Phase 2: Explosive extension
        t = (jump_elapsed - JUMP_CROUCH_TIME) / JUMP_PUSH_TIME
        return {
            'hip': 0.0,
            'thigh': 1.4 - t * 1.2,  # 1.4 -> 0.2 (extend)
            'calf': -2.8 + t * 2.4   # -2.8 -> -0.4 (extend)
        }
    else:
        # Phase 3: Flight - legs slightly tucked
        return {
            'hip': 0.0,
            'thigh': 0.5,
            'calf': -1.0
        }

def controller(model, data):
    """PD controller called at each simulation step"""
    global current_pose, target_pose, transition_start_time
    global is_spinning, spin_start_time
    global is_jumping, jump_start_time
    
    # Handle jumping (overrides other motions)
    if is_jumping:
        jump_elapsed = data.time - jump_start_time
        
        if jump_elapsed >= JUMP_TOTAL_TIME:
            # Jump complete, return to standing
            is_jumping = False
            print(f"Jump complete! (time={data.time:.2f})")
        
        jump_pose = get_jump_pose(jump_elapsed)
        
        # Build targets for jump
        TARGETS = {}
        for act_idx in range(12):
            joint_type = ACT_TO_JOINT[act_idx]
            TARGETS[act_idx] = jump_pose[joint_type]
        
        # Apply PD control
        for act_idx in range(12):
            target = TARGETS[act_idx]
            qpos_idx = ACT_TO_QPOS[act_idx]
            qvel_idx = ACT_TO_QVEL[act_idx]
            
            pos = data.qpos[qpos_idx]
            vel = data.qvel[qvel_idx]
            
            # Higher gains for explosive jump
            torque = (KP * 1.5) * (target - pos) - KD * vel
            torque = np.clip(torque, -23.0, 23.0)
            
            data.ctrl[act_idx] = torque
        
        # Wheels off
        for i in range(12, 16):
            data.ctrl[i] = 0.0
        
        return  # Skip rest of controller
    
    # Determine current target positions
    if transition_start_time is not None:
        elapsed = data.time - transition_start_time
        t = elapsed / TRANSITION_DURATION
        
        if t >= 1.0:
            current_pose = target_pose
            transition_start_time = None
            print(f"Transition complete: now {current_pose}")
            t = 1.0
        
        pose = interpolate_pose(POSES[current_pose], POSES[target_pose], t)
    else:
        pose = POSES[current_pose]
    
    # Calculate spin phase if spinning
    spin_phase = 0.0
    if is_spinning:
        spin_elapsed = data.time - spin_start_time
        spin_phase = (spin_elapsed % SPIN_CYCLE_TIME) / SPIN_CYCLE_TIME
    
    # Build targets
    TARGETS = {}
    for act_idx in range(12):
        joint_type = ACT_TO_JOINT[act_idx]
        base_target = pose[joint_type]
        
        # Add spin motion to hip joints
        if is_spinning and joint_type == 'hip':
            hip_offset = get_spin_hip_offset(act_idx, spin_phase)
            TARGETS[act_idx] = base_target + hip_offset
        else:
            TARGETS[act_idx] = base_target
    
    # PD control
    for act_idx in range(12):
        target = TARGETS[act_idx]
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
    global target_pose, transition_start_time, current_pose
    global is_spinning, spin_start_time
    global is_jumping, jump_start_time
    
    # J key triggers jump
    if keycode in [74, 106]:  # J or j
        if not is_jumping and not transition_start_time:
            is_jumping = True
            is_spinning = False  # Stop spinning if active
            jump_start_time = data.time
            print(f"Command: JUMP! (time={data.time:.2f})")
        return
    
    # Don't allow other commands during jump
    if is_jumping:
        return
    
    # S key toggles spin
    if keycode in [83, 115]:  # S or s
        is_spinning = not is_spinning
        if is_spinning:
            spin_start_time = data.time
            # Transition to crouch pose for better stability
            if current_pose != 'spin_crouch':
                target_pose = 'spin_crouch'
                transition_start_time = data.time
                print(f"Command: Spinning ON - crouching... (time={data.time:.2f})")
            else:
                print(f"Command: Spinning ON (time={data.time:.2f})")
        else:
            print(f"Command: Spinning OFF (time={data.time:.2f})")
        return
    
    if transition_start_time is not None:
        return  # Already transitioning
    
    # Other pose controls
    if keycode in [85, 117]:  # U or u
        if current_pose != 'standing':
            target_pose = 'standing'
            transition_start_time = data.time
            is_spinning = False  # Stop spinning when changing pose
            print(f"Command: Standing up... (time={data.time:.2f})")
    elif keycode in [73, 105]:  # I or i
        if current_pose != 'sitting':
            target_pose = 'sitting'
            transition_start_time = data.time
            is_spinning = False
            print(f"Command: Sitting down... (time={data.time:.2f})")
    elif keycode in [76, 108]:  # L or l
        if current_pose != 'lying':
            target_pose = 'lying'
            transition_start_time = data.time
            is_spinning = False
            print(f"Command: Laying down... (time={data.time:.2f})")

print("=" * 60)
print("REALISTIC SPIN & JUMP CONTROL")
print("=" * 60)
print("Press keys while the viewer window has focus:")
print("  J - Jump (crouch -> explosive push -> flight)")
print("  S - Toggle spin (legs paddle to create torque)")
print("  U - Stand up")
print("  I - Sit down")
print("  L - Lay down")
print("\nThe spin uses coordinated leg paddling:")
print("  - Front-right & rear-left push one way")
print("  - Front-left & rear-right push the other")
print("  - Creates rotational torque through ground contact")
print("\nThe jump uses realistic three-phase motion:")
print("  - Phase 1: Crouch down (load energy)")
print("  - Phase 2: Explosive extension (push off)")
print("  - Phase 3: Flight with legs tucked")
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