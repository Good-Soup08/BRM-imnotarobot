# for testing only: input (x,y,z) and evaluate robot control, using the IK algorithm

import pyfirmata
import time
import os
import numpy as np
import math
from app import Gestures

l1 = 6.1    # Shoulder → elbow
l2 = 11.5   # Elbow → end effector
h  = 6.8    # Shoulder height above ground

def inverse_kinematics_point(x, y, z, elbow_up=False):

    # 1) base yaw:
    phi = math.atan2(y, x)
    theta1_deg = math.degrees(phi + math.pi/2) # map to robot angles

    # 2) project into the shoulder's pitch plane:
    r = math.hypot(x, y) # horizontal distance from shoulder
    dz = z - h # vertical offset from shoulder

    # 3) two‑link planar IK:
    D = (r**2 + dz**2 - l1**2 - l2**2) / (2*l1*l2)
    if abs(D) > 1.0:
        raise ValueError(f"Point ({x}, {y}, {z}) is out of reach (|D|={D:.3f}>1).") # limitation of the arm length

    # 4) elbow angle:
    if elbow_up:
        theta3_rad = math.atan2(+math.sqrt(1-D**2), D)
    else:
        theta3_rad = math.atan2(-math.sqrt(1-D**2), D)

    # 5) shoulder pitch:
    alpha = math.atan2(dz, r)
    beta = math.atan2(l2 * math.sin(theta3_rad), l1 + l2 * math.cos(theta3_rad))
    theta2_rad = alpha - beta

    # 6) map to robot angle conventions:
    theta2_deg = math.degrees(math.pi - theta2_rad)
    theta3_deg = math.degrees(theta3_rad + math.pi)

    return theta1_deg, theta2_deg, theta3_deg

# connect to the Arduino board on COM3 and setup pins:
board = pyfirmata.Arduino("COM3")
it = pyfirmata.util.Iterator(board)
it.start()
servo_pins = [
    board.get_pin('d:4:s'),  # Servo 1 on pin 4
    board.get_pin('d:5:s'),  # Servo 2 on pin 5
    board.get_pin('d:6:s'),  # Servo 3 on pin 6
]

# board initialize time:
time.sleep(1)

# take inputs from keyboard:
x = float(input("x?"))
y = float(input("y?"))
z = float(input("z?"))

angles = inverse_kinematics_point(x, y, z, elbow_up=True)

# apply to robot:
for i, servo in enumerate(servo_pins):
    servo.write(angles[i])
    time.sleep(0.015)
    
time.sleep(1)
board.sp.close()
