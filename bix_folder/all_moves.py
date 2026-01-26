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

# Movement state
is_moving = False
move_direction = 'forward'  # 'forward' or 'backward'

# === MOVEMENT TUNING PARAMETERS ===
MOVE_WHEEL_SPEED = 10.0  # Wheel rotation speed (rad/s) - adjust for faster/slower movement

# Jump state
is_jumping = False
jump_start_time = 0.0

# === JUMP TUNING PARAMETERS ===
# Adjust these if the jump doesn't look right in your simulation
JUMP_CROUCH_TIME = 0.3   # Time to crouch before jump (increase if needs more windup)
JUMP_PUSH_TIME = 0.15    # Time to push off ground (decrease for more explosive)
JUMP_FLIGHT_TIME = 0.4   # Time in air (adjust based on actual hang time)
JUMP_LAND_TIME = 0.3     # Time to absorb landing (increase if falling over)

# FRONT LEGS - Jump pose angles
JUMP_FRONT_CROUCH_THIGH = 1.4   # How much to bend thighs in crouch (higher = more crouch)
JUMP_FRONT_CROUCH_CALF = -2.8   # How much to bend calves in crouch (more negative = more crouch)
JUMP_FRONT_PUSH_THIGH = 0.2     # Thigh angle during push (lower = more extension)
JUMP_FRONT_PUSH_CALF = -0.4     # Calf angle during push (closer to 0 = more extension)
JUMP_FRONT_FLIGHT_THIGH = 0.5   # Thigh angle in flight (adjust for tucking)
JUMP_FRONT_FLIGHT_CALF = -1.0   # Calf angle in flight (adjust for tucking)
JUMP_FRONT_LAND_THIGH = 0.8     # Final thigh angle after landing
JUMP_FRONT_LAND_CALF = -1.6     # Final calf angle after landing

# REAR LEGS - Jump pose angles (adjust these to fix back-flip problem)
JUMP_REAR_CROUCH_THIGH = 1.2    # Less crouch than front to balance jump
JUMP_REAR_CROUCH_CALF = -2.4    # Less crouch than front to balance jump
JUMP_REAR_PUSH_THIGH = 0.1      # More extension than front (push harder)
JUMP_REAR_PUSH_CALF = -0.2      # More extension than front (push harder)
JUMP_REAR_FLIGHT_THIGH = 0.5   
JUMP_REAR_FLIGHT_CALF = -1.0   
JUMP_REAR_LAND_THIGH = 0.8     
JUMP_REAR_LAND_CALF = -1.6     

JUMP_TORQUE_MULTIPLIER = 1.5  # Torque boost during push (increase for higher jump)

JUMP_TOTAL_TIME = JUMP_CROUCH_TIME + JUMP_PUSH_TIME + JUMP_FLIGHT_TIME + JUMP_LAND_TIME

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

def get_jump_pose(jump_elapsed, is_rear_leg=False):
    """
    Calculate pose for jumping motion
    is_rear_leg: True for rear legs, False for front legs
    Returns dict with thigh/calf angles
    """
    # Select angle set based on leg position
    if is_rear_leg:
        crouch_thigh = JUMP_REAR_CROUCH_THIGH
        crouch_calf = JUMP_REAR_CROUCH_CALF
        push_thigh = JUMP_REAR_PUSH_THIGH
        push_calf = JUMP_REAR_PUSH_CALF
        flight_thigh = JUMP_REAR_FLIGHT_THIGH
        flight_calf = JUMP_REAR_FLIGHT_CALF
        land_thigh = JUMP_REAR_LAND_THIGH
        land_calf = JUMP_REAR_LAND_CALF
    else:
        crouch_thigh = JUMP_FRONT_CROUCH_THIGH
        crouch_calf = JUMP_FRONT_CROUCH_CALF
        push_thigh = JUMP_FRONT_PUSH_THIGH
        push_calf = JUMP_FRONT_PUSH_CALF
        flight_thigh = JUMP_FRONT_FLIGHT_THIGH
        flight_calf = JUMP_FRONT_FLIGHT_CALF
        land_thigh = JUMP_FRONT_LAND_THIGH
        land_calf = JUMP_FRONT_LAND_CALF
    
    if jump_elapsed < JUMP_CROUCH_TIME:
        # Phase 1: Crouch down (load energy)
        t = jump_elapsed / JUMP_CROUCH_TIME
        return {
            'hip': 0.0,
            'thigh': 0.8 + t * (crouch_thigh - 0.8),
            'calf': -1.6 + t * (crouch_calf - (-1.6))
        }
    elif jump_elapsed < JUMP_CROUCH_TIME + JUMP_PUSH_TIME:
        # Phase 2: Explosive extension (push off)
        t = (jump_elapsed - JUMP_CROUCH_TIME) / JUMP_PUSH_TIME
        return {
            'hip': 0.0,
            'thigh': crouch_thigh + t * (push_thigh - crouch_thigh),
            'calf': crouch_calf + t * (push_calf - crouch_calf)
        }
    elif jump_elapsed < JUMP_CROUCH_TIME + JUMP_PUSH_TIME + JUMP_FLIGHT_TIME:
        # Phase 3: Flight - legs tucked
        return {
            'hip': 0.0,
            'thigh': flight_thigh,
            'calf': flight_calf
        }
    else:
        # Phase 4: Landing absorption - bend legs to cushion impact
        t = (jump_elapsed - JUMP_CROUCH_TIME - JUMP_PUSH_TIME - JUMP_FLIGHT_TIME) / JUMP_LAND_TIME
        return {
            'hip': 0.0,
            'thigh': flight_thigh + t * (land_thigh - flight_thigh),
            'calf': flight_calf + t * (land_calf - flight_calf)
        }

def controller(model, data):
    """PD controller called at each simulation step"""
    global current_pose, target_pose, transition_start_time
    global is_spinning, spin_start_time
    global is_jumping, jump_start_time
    global is_moving, move_direction
    
    # Handle jumping (overrides other motions)
    if is_jumping:
        jump_elapsed = data.time - jump_start_time
        
        if jump_elapsed >= JUMP_TOTAL_TIME:
            # Jump complete, return to standing
            is_jumping = False
            print(f"Jump complete! (time={data.time:.2f})")
        
        # Build targets for jump (different for front vs rear legs)
        TARGETS = {}
        for act_idx in range(12):
            joint_type = ACT_TO_JOINT[act_idx]
            is_rear_leg = act_idx >= 6  # RR and RL are actuators 6-11
            jump_pose = get_jump_pose(jump_elapsed, is_rear_leg)
            TARGETS[act_idx] = jump_pose[joint_type]
        
        # Apply PD control
        for act_idx in range(12):
            target = TARGETS[act_idx]
            qpos_idx = ACT_TO_QPOS[act_idx]
            qvel_idx = ACT_TO_QVEL[act_idx]
            
            pos = data.qpos[qpos_idx]
            vel = data.qvel[qvel_idx]
            
            # Use torque multiplier during push phase for explosive jump
            jump_elapsed = data.time - jump_start_time
            if JUMP_CROUCH_TIME <= jump_elapsed < JUMP_CROUCH_TIME + JUMP_PUSH_TIME:
                torque = (KP * JUMP_TORQUE_MULTIPLIER) * (target - pos) - KD * vel
            else:
                torque = KP * (target - pos) - KD * vel
            
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
    
    # Wheels control (actuators 12-15)
    if is_moving:
        # Direction multiplier
        dir_mult = 1.0 if move_direction == 'forward' else -1.0
        # Spin all wheels for forward/backward movement
        for i in range(12, 16):
            data.ctrl[i] = dir_mult * MOVE_WHEEL_SPEED
    else:
        # Wheels off
        for i in range(12, 16):
            data.ctrl[i] = 0.0

def key_callback(keycode):
    """Handle keyboard input"""
    global target_pose, transition_start_time, current_pose
    global is_spinning, spin_start_time
    global is_jumping, jump_start_time
    global is_moving, move_direction
    
    # J key triggers jump
    if keycode in [74, 106]:  # J or j
        if not is_jumping and not transition_start_time:
            is_jumping = True
            is_spinning = False  # Stop spinning if active
            is_moving = False    # Stop moving if active
            jump_start_time = data.time
            print(f"Command: JUMP! (time={data.time:.2f})")
        return
    
    # Don't allow other commands during jump
    if is_jumping:
        return
    
    # F key - move forward
    if keycode in [70, 102]:  # F or f
        is_moving = not is_moving
        if is_moving:
            move_direction = 'forward'
            is_spinning = False  # Stop spinning if active
            print(f"Command: Wheels FORWARD (time={data.time:.2f})")
        else:
            print(f"Command: Wheels stopped (time={data.time:.2f})")
        return
    
    # K key - move backward
    if keycode in [75, 107]:  # K or k
        is_moving = not is_moving
        if is_moving:
            move_direction = 'backward'
            is_spinning = False  # Stop spinning if active
            print(f"Command: Wheels BACKWARD (time={data.time:.2f})")
        else:
            print(f"Command: Wheels stopped (time={data.time:.2f})")
        return
    
    # S key toggles spin
    if keycode in [83, 115]:  # S or s
        is_spinning = not is_spinning
        if is_spinning:
            spin_start_time = data.time
            is_moving = False  # Stop moving if active
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
            is_moving = False    # Stop moving when changing pose
            print(f"Command: Standing up... (time={data.time:.2f})")
    elif keycode in [73, 105]:  # I or i
        if current_pose != 'sitting':
            target_pose = 'sitting'
            transition_start_time = data.time
            is_spinning = False
            is_moving = False
            print(f"Command: Sitting down... (time={data.time:.2f})")
    elif keycode in [76, 108]:  # L or l
        if current_pose != 'lying':
            target_pose = 'lying'
            transition_start_time = data.time
            is_spinning = False
            is_moving = False
            print(f"Command: Laying down... (time={data.time:.2f})")

print("=" * 60)
print("GO2-W: SPIN, JUMP & WHEEL MOVEMENT CONTROL")
print("=" * 60)
print("Press keys while the viewer window has focus:")
print("  J - Jump (crouch -> explosive push -> flight -> land)")
print("  F - Toggle forward wheel movement")
print("  K - Toggle backward wheel movement")
print("  S - Toggle spin (legs paddle to create torque)")
print("  U - Stand up")
print("  I - Sit down")
print("  L - Lay down")
print("\nWheel movement:")
print("  - All 4 wheels spin together for forward/backward motion")
print("  - Adjust MOVE_WHEEL_SPEED to control speed")
print("\nThe spin uses coordinated leg paddling:")
print("  - Front-right & rear-left push one way")
print("  - Front-left & rear-right push the other")
print("  - Creates rotational torque through ground contact")
print("\nThe jump uses realistic four-phase motion:")
print("  - Phase 1: Crouch down (load energy)")
print("  - Phase 2: Explosive extension (push off)")
print("  - Phase 3: Flight with legs tucked")
print("  - Phase 4: Landing absorption (cushion impact)")
print("\nJUMP TUNING: Edit these parameters at top of file if needed:")
print("  - JUMP_*_TIME: Adjust phase durations")
print("  - JUMP_FRONT_*: Front leg angles (if jumping too much forward)")
print("  - JUMP_REAR_*: Rear leg angles (if falling backward/forward)")
print("  - JUMP_TORQUE_MULTIPLIER: Control jump height")
print("\nMOVEMENT TUNING:")
print("  - MOVE_WHEEL_SPEED: Wheel rotation speed")
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