#!/usr/bin/env python3
# ============================================
# GO2-W FRONT WHEELIE - COMPLETE CODE
# ============================================
# 
# Run this from terminal with:
#   mjpython wheelie_go2w.py
#
# Controls in viewer:
#   - Mouse drag: Rotate camera
#   - Scroll: Zoom
#   - Space: Pause/Resume
#   - ESC: Close
# ============================================

import mujoco
import mujoco.viewer
import time
import numpy as np
import os

# ============================================
# 1. LOAD MODEL
# ============================================

# Try different paths
possible_paths = [
    'unitree_mujoco/unitree_robots/go2w/scene.xml',
    '../unitree_mujoco/unitree_robots/go2w/scene.xml',
    './unitree_robots/go2w/scene.xml',
    os.path.expanduser('~/unitree_mujoco/unitree_robots/go2w/scene.xml'),
]

model_path = None
for path in possible_paths:
    if os.path.exists(path):
        model_path = path
        break

if model_path is None:
    print("=" * 50)
    print("ERROR: Go2-W model not found!")
    print("=" * 50)
    print("\nPlease clone unitree_mujoco first:")
    print("  git clone https://github.com/unitreerobotics/unitree_mujoco.git")
    print("\nThen run this script from the same directory.")
    exit(1)

print("=" * 50)
print("GO2-W FRONT WHEELIE SIMULATION")
print("=" * 50)
print(f"\n✓ Loading model from: {model_path}")

model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

print(f"✓ Model loaded!")
print(f"  - Joints: {model.njnt}")
print(f"  - Actuators: {model.nu}")

# ============================================
# 2. PRINT ACTUATOR INFO
# ============================================

print("\n--- Actuators ---")
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"  [{i:2d}] {name}")

# ============================================
# 3. CONFIGURE ROBOT POSE
# ============================================

# Reset simulation
mujoco.mj_resetData(model, data)

# --- Base position and orientation ---
# qpos[0:3] = x, y, z position
# qpos[3:7] = quaternion (w, x, y, z)

data.qpos[0] = 0.0    # X
data.qpos[1] = 0.0    # Y
data.qpos[2] = 0.45   # Z height - adjust if robot floats or clips ground

# Tilt forward for front wheelie
# Quaternion: rotate around Y axis to tilt forward
tilt_angle = 0.3  # radians (~17 degrees forward tilt)
data.qpos[3] = np.cos(tilt_angle / 2)  # w
data.qpos[4] = 0.0                      # x
data.qpos[5] = np.sin(tilt_angle / 2)   # y (tilt forward)
data.qpos[6] = 0.0                      # z

# --- Leg joint positions ---
# Go2-W has 12 leg joints (3 per leg) + 4 wheel joints
# 
# Joint order (typical):
#   0-2:  Front Right (hip, thigh, calf)
#   3-5:  Front Left
#   6-8:  Rear Right
#   9-11: Rear Left
#   12-15: Wheels (if in qpos)

WHEELIE_POSE = [
    # Front Right - extended down to support robot
    0.0,    # hip abduction
    0.5,    # thigh forward
    -1.0,   # calf (knee bent)
    
    # Front Left - extended down to support robot
    0.0,
    0.5,
    -1.0,
    
    # Rear Right - tucked up (off the ground)
    0.0,
    1.8,    # thigh pulled way back
    -2.8,   # calf fully tucked
    
    # Rear Left - tucked up (off the ground)
    0.0,
    1.8,
    -2.8,
]

# Apply pose to joints (qpos[7:19] are leg joints)
for i, angle in enumerate(WHEELIE_POSE):
    if 7 + i < len(data.qpos):
        data.qpos[7 + i] = angle

# Zero initial velocities
data.qvel[:] = 0.0

# Update physics state
mujoco.mj_forward(model, data)

print(f"\n✓ Robot spawned in wheelie position")
print(f"  - Height: {data.qpos[2]:.2f} m")
print(f"  - Tilt: {np.degrees(tilt_angle):.1f} degrees forward")

# ============================================
# 4. CONTROL PARAMETERS
# ============================================

# PD control gains for legs
KP = 80.0   # Position gain (stiffness)
KD = 4.0    # Velocity gain (damping)

# Wheel actuator indices (adjust if different for your model)
FRONT_RIGHT_WHEEL = 12
FRONT_LEFT_WHEEL = 13
REAR_RIGHT_WHEEL = 14
REAR_LEFT_WHEEL = 15

# Wheel speed
WHEEL_SPEED = 12.0  # Increase for faster movement

print(f"\n--- Control Settings ---")
print(f"  KP: {KP}, KD: {KD}")
print(f"  Wheel speed: {WHEEL_SPEED}")
print(f"  Front wheels: [{FRONT_RIGHT_WHEEL}, {FRONT_LEFT_WHEEL}]")

# ============================================
# 5. RUN SIMULATION WITH VIEWER
# ============================================

print("\n" + "=" * 50)
print("STARTING SIMULATION")
print("=" * 50)
print("\nControls:")
print("  - Mouse drag: Rotate camera")
print("  - Scroll: Zoom in/out")
print("  - Space: Pause/Resume")
print("  - Right arrow: Step (when paused)")
print("  - ESC: Close viewer")
print("\nThe robot will:")
print("  1. Balance on front wheels")
print("  2. Start driving forward after 2 seconds")
print("=" * 50)

with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    
    while viewer.is_running():
        current_time = time.time() - start_time
        
        # === LEG CONTROL ===
        # Use PD control to maintain the wheelie pose
        for i in range(12):
            if i < len(WHEELIE_POSE):
                target_pos = WHEELIE_POSE[i]
                current_pos = data.qpos[7 + i]
                current_vel = data.qvel[6 + i]
                
                # PD control: torque = Kp*(target - current) - Kd*velocity
                torque = KP * (target_pos - current_pos) - KD * current_vel
                data.ctrl[i] = torque
        
        # === WHEEL CONTROL ===
        if model.nu > 12:  # Make sure we have wheel actuators
            
            # Wait 2 seconds before starting to drive
            if current_time > 2.0:
                # Drive forward on front wheels
                data.ctrl[FRONT_RIGHT_WHEEL] = WHEEL_SPEED
                data.ctrl[FRONT_LEFT_WHEEL] = WHEEL_SPEED
                
                # === OPTIONAL: Different movements ===
                # Uncomment one of these for different behaviors:
                
                # Turn LEFT (right wheel faster):
                # data.ctrl[FRONT_RIGHT_WHEEL] = WHEEL_SPEED * 1.5
                # data.ctrl[FRONT_LEFT_WHEEL] = WHEEL_SPEED * 0.5
                
                # Turn RIGHT (left wheel faster):
                # data.ctrl[FRONT_RIGHT_WHEEL] = WHEEL_SPEED * 0.5
                # data.ctrl[FRONT_LEFT_WHEEL] = WHEEL_SPEED * 1.5
                
                # SPIN in place:
                # data.ctrl[FRONT_RIGHT_WHEEL] = WHEEL_SPEED
                # data.ctrl[FRONT_LEFT_WHEEL] = -WHEEL_SPEED
                
                # Drive BACKWARD:
                # data.ctrl[FRONT_RIGHT_WHEEL] = -WHEEL_SPEED
                # data.ctrl[FRONT_LEFT_WHEEL] = -WHEEL_SPEED
                
            else:
                # Keep wheels still during initial balance
                data.ctrl[FRONT_RIGHT_WHEEL] = 0.0
                data.ctrl[FRONT_LEFT_WHEEL] = 0.0
            
            # Always keep rear wheels off
            data.ctrl[REAR_RIGHT_WHEEL] = 0.0
            data.ctrl[REAR_LEFT_WHEEL] = 0.0
        
        # Step the simulation
        mujoco.mj_step(model, data)
        
        # Sync the viewer
        viewer.sync()

print("\n✓ Simulation ended!")
print("=" * 50)
