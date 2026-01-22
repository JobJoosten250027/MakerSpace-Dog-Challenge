#!/usr/bin/env python3
"""
GO2-W STANDING - CORRECT JOINT ORDER
"""

#Importing necessary stuff
import mujoco
import mujoco.viewer
import time
import numpy as np

#Import and load the model of the Unitree Go2-W
model = mujoco.MjModel.from_xml_path('unitree_mujoco/unitree_robots/go2w/scene.xml')
data = mujoco.MjData(model)

#Completely resets simulation to zero state
mujoco.mj_resetData(model, data)

# Position (Base position of where robot is set in simulation; qpos 0-2 are center posisitions in world coordinates (XYZ).)
data.qpos[0] = 0.0   # X
data.qpos[1] = 0.0   # Y
data.qpos[2] = 0.45  # Z

# Upright orientation
data.qpos[3] = 1.0
data.qpos[4] = 0.0
data.qpos[5] = 0.0
data.qpos[6] = 0.0

# CORRECT JOINT ORDER:
# qpos[7-10]:  FL (hip, thigh, calf, wheel)
# qpos[11-14]: FR (hip, thigh, calf, wheel)
# qpos[15-18]: RL (hip, thigh, calf, wheel)
# qpos[19-22]: RR (hip, thigh, calf, wheel)

# Standing pose: hip=0, thigh=0.8, calf=-1.6, wheel=0
data.qpos[7]  = 0.0    # FL hip
data.qpos[8]  = 0.8    # FL thigh
data.qpos[9]  = -1.6   # FL calf
data.qpos[10] = 0.0    # FL wheel

data.qpos[11] = 0.0    # FR hip
data.qpos[12] = 0.8    # FR thigh
data.qpos[13] = -1.6   # FR calf
data.qpos[14] = 0.0    # FR wheel

data.qpos[15] = 0.0    # RL hip
data.qpos[16] = 0.8    # RL thigh
data.qpos[17] = -1.6   # RL calf
data.qpos[18] = 0.0    # RL wheel

data.qpos[19] = 0.0    # RR hip
data.qpos[20] = 0.8    # RR thigh
data.qpos[21] = -1.6   # RR calf
data.qpos[22] = 0.0    # RR wheel

data.qvel[:] = 0
mujoco.mj_forward(model, data)

# Target positions for control (same order as actuators)
# Actuator order: FR, FL, RR, RL (different from qpos!)
# [0-2] FR hip,thigh,calf  [3-5] FL  [6-8] RR  [9-11] RL  [12-15] wheels

TARGETS = {
    # FR
    0: 0.0,   1: 0.8,   2: -1.6,
    # FL  
    3: 0.0,   4: 0.8,   5: -1.6,
    # RR
    6: 0.0,   7: 0.8,   8: -1.6,
    # RL
    9: 0.0,  10: 0.8,  11: -1.6,
}

# Map actuator index to qpos index
ACT_TO_QPOS = {
    0: 11,  1: 12,  2: 13,   # FR hip, thigh, calf -> qpos 11,12,13
    3: 7,   4: 8,   5: 9,    # FL hip, thigh, calf -> qpos 7,8,9
    6: 19,  7: 20,  8: 21,   # RR hip, thigh, calf -> qpos 19,20,21
    9: 15, 10: 16, 11: 17,   # RL hip, thigh, calf -> qpos 15,16,17
}

# Map actuator index to qvel index (qvel is nv, slightly different)
ACT_TO_QVEL = {
    0: 10,  1: 11,  2: 12,   # FR
    3: 6,   4: 7,   5: 8,    # FL
    6: 18,  7: 19,  8: 20,   # RR
    9: 14, 10: 15, 11: 16,   # RL
}

KP = 500.0
KD = 20.0

print(f"KP={KP}, KD={KD}")
print("Press ESC to exit")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        
        # PD control for leg actuators
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