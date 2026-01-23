#!/usr/bin/env python3
"""
GO2-W STANDING UP / LAYING DOWN - MATPLOTLIB CONTROL
"""

import mujoco
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for Windows
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

KP = 500.0
KD = 20.0

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

print("Initializing renderer...")
# Initialize renderer
renderer = mujoco.Renderer(model, height=480, width=640)

# Start in standing pose
set_initial_pose('standing')

# Animation state
current_pose = 'standing'
target_pose = 'standing'
transition_start_time = None
TRANSITION_DURATION = 2.0
sim_time = 0.0

print("Creating window...")
# Create figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
plt.subplots_adjust(bottom=0.2)

# Image display
img_display = ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
ax.axis('off')
ax.set_title(f'Current Pose: {current_pose}', fontsize=14, fontweight='bold')

# Keyboard control
def on_key(event):
    global target_pose, transition_start_time
    
    print(f"Key pressed: {event.key}")  # Debug
    
    if transition_start_time is not None:
        print("Already transitioning, ignoring key press")
        return  # Already transitioning
    
    if event.key == 'u' and current_pose != 'standing':
        target_pose = 'standing'
        transition_start_time = sim_time
        print(f"Command: Standing up... (sim_time={sim_time})")
    elif event.key == 'i' and current_pose != 'sitting':
        target_pose = 'sitting'
        transition_start_time = sim_time
        print(f"Command: Sitting down... (sim_time={sim_time})")
    elif event.key == 'l' and current_pose != 'lying':
        target_pose = 'lying'
        transition_start_time = sim_time
        print(f"Command: Laying down... (sim_time={sim_time})")

fig.canvas.mpl_connect('key_press_event', on_key)

def simulate_step():
    """Run one simulation step"""
    global current_pose, transition_start_time, sim_time
    
    # Determine current target positions
    if transition_start_time is not None:
        elapsed = sim_time - transition_start_time
        t = elapsed / TRANSITION_DURATION
        
        if t >= 1.0:
            current_pose = target_pose
            transition_start_time = None
            print(f"Transition complete: now {current_pose}")
            t = 1.0
        
        pose = interpolate_pose(POSES[current_pose], POSES[target_pose], t)
    else:
        pose = POSES[current_pose]
    
    # Build targets
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
    
    # Step simulation
    mujoco.mj_step(model, data)
    sim_time += model.opt.timestep

def update_frame(frame):
    """Update animation frame"""
    global current_pose, transition_start_time
    
    # Run multiple simulation steps per frame for smoother physics
    for _ in range(5):
        simulate_step()
    
    # Render
    renderer.update_scene(data)
    pixels = renderer.render()
    
    # Update display
    img_display.set_array(pixels)
    
    # Update title based on current state
    if transition_start_time is not None:
        elapsed = sim_time - transition_start_time
        t = elapsed / TRANSITION_DURATION
        progress = int(t * 100)
        ax.set_title(f'Transitioning to {target_pose}: {progress}%', fontsize=14, fontweight='bold')
    else:
        ax.set_title(f'Current Pose: {current_pose}', fontsize=14, fontweight='bold')
    
    return [img_display]

print("Starting animation...")
# Create animation without blit to avoid animation errors
ani = animation.FuncAnimation(fig, update_frame, interval=20, blit=False, cache_frame_data=False)

print("=" * 60)
print("KEYBOARD CONTROL")
print("=" * 60)
print("The window should now be visible!")
print("Make sure the matplotlib window has focus, then press:")
print("  U - Stand up")
print("  I - Sit down")
print("  L - Lay down")
print("\nClose the window to exit")
print("=" * 60)

plt.show()

print("Done!")