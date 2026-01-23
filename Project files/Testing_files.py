#!/usr/bin/env python3
"""
GO2-W STANDING UP / LAYING DOWN - MANUAL CONTROL
"""

import mujoco
import mujoco.viewer
import time
import numpy as np
import threading

model = mujoco.MjModel.from_xml_path('unitree_mujoco/unitree_robots/go2w/scene.xml')
data = mujoco.MjData(model)

mujoco.mj_resetData(model, data)

# Initial position
data.qpos[0] = 0.0   # X
data.qpos[1] = 0.0   # Y
data.qpos[2] = 0.45  # Z

# Upright orientation
data.qpos[3] = 1.0
data.qpos[4] = 0.0
data.qpos[5] = 0.0
data.qpos[6] = 0.0

# Define poses (hip, thigh, calf for each leg)
POSES = {
    'lying': {
        'z': 0.15,
        'hip': 0.0,
        'thigh': 1.4,
        'calf': -2.8
    },
    'standing': {
        'z': 0.45,
        'hip': 0.0,
        'thigh': 0.8,
        'calf': -1.6
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

# Map actuator to joint type
ACT_TO_JOINT = {
    0: 'hip',   1: 'thigh',  2: 'calf',   # FR
    3: 'hip',   4: 'thigh',  5: 'calf',   # FL
    6: 'hip',   7: 'thigh',  8: 'calf',   # RR
    9: 'hip',  10: 'thigh', 11: 'calf',   # RL
}

KP = 500.0
KD = 20.0

def interpolate_pose(pose_start, pose_end, t):
    """Interpolate between two poses. t ranges from 0 (start) to 1 (end)"""
    t = np.clip(t, 0.0, 1.0)
    
    pose = {}
    for key in pose_start:
        pose[key] = pose_start[key] + t * (pose_end[key] - pose_start[key])
    
    return pose

def set_initial_pose(pose_name):
    """Set robot to initial pose without simulation"""
    pose = POSES[pose_name]
    
    data.qpos[2] = pose['z']
    
    # Set all legs
    for leg_start in [7, 11, 15, 19]:  # FL, FR, RL, RR starting indices
        data.qpos[leg_start]     = pose['hip']
        data.qpos[leg_start + 1] = pose['thigh']
        data.qpos[leg_start + 2] = pose['calf']
        data.qpos[leg_start + 3] = 0.0  # wheel
    
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

# Start in lying pose
set_initial_pose('lying')

print(f"KP={KP}, KD={KD}")
print("\n" + "="*50)
print("CONTROLS - Type in terminal and press ENTER:")
print("  s - Stand up")
print("  l - Lay down")
print("  q - Quit")
print("="*50 + "\n")

# Animation state
current_pose = 'lying'
target_pose = 'lying'
transition_start_time = None
TRANSITION_DURATION = 2.0  # seconds

# Command queue
command_queue = []
command_lock = threading.Lock()
running = True

def keyboard_listener():
    """Listen for keyboard input in a separate thread"""
    global running
    while running:
        try:
            user_input = input("Enter command (s/l/q): ").strip().lower()
            with command_lock:
                if user_input == 's':
                    command_queue.append('stand')
                    print("-> Command received: Stand up")
                elif user_input == 'l':
                    command_queue.append('lie')
                    print("-> Command received: Lay down")
                elif user_input == 'q':
                    command_queue.append('quit')
                    running = False
                    print("-> Quitting...")
                else:
                    print("-> Invalid command. Use s, l, or q")
        except EOFError:
            break
        except Exception as e:
            print(f"Error: {e}")
            break

# Start keyboard listener thread
kbd_thread = threading.Thread(target=keyboard_listener, daemon=True)
kbd_thread.start()

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running() and running:
        
        # Process commands
        with command_lock:
            while command_queue:
                cmd = command_queue.pop(0)
                
                if cmd == 'quit':
                    running = False
                    break
                
                # Only start new transition if not already transitioning
                if transition_start_time is None:
                    if cmd == 'stand' and current_pose != 'standing':
                        target_pose = 'standing'
                        transition_start_time = data.time
                        print(">>> Standing up...")
                    elif cmd == 'lie' and current_pose != 'lying':
                        target_pose = 'lying'
                        transition_start_time = data.time
                        print(">>> Laying down...")
        
        # Determine current target positions
        if transition_start_time is not None:
            elapsed = data.time - transition_start_time
            t = elapsed / TRANSITION_DURATION
            
            if t >= 1.0:
                # Transition complete
                current_pose = target_pose
                transition_start_time = None
                print(f">>> Transition complete: now {current_pose}")
                t = 1.0
            
            pose = interpolate_pose(POSES[current_pose], POSES[target_pose], t)
        else:
            pose = POSES[current_pose]
        
        # Build targets dictionary
        TARGETS = {}
        for act_idx in range(12):
            joint_type = ACT_TO_JOINT[act_idx]
            TARGETS[act_idx] = pose[joint_type]
        
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
        
        mujoco.mj_step(model, data)
        viewer.sync()

print("Done!")