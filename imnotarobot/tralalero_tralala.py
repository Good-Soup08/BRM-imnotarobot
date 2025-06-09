import pyfirmata
import time
import os
import numpy as np
import math
from app import Gestures

# Link lengths and shoulder height (same as in forward_kinematics_points)
L1 = 6.1   # Shoulder → elbow
L2 = 7.0   # Elbow → end effector
H  = 6.8   # Shoulder height above ground


def inverse_kinematics_point(x, y, z, elbow_up=False): # ta ao contrario o booleano
    """
    Compute inverse kinematics for the 3-DOF arm.

    Args:
        x, y, z: Desired end-effector coordinates.
        elbow_up: If True, returns the "elbow-up" solution; otherwise "elbow-down".

    Returns:
        (theta1_deg, theta2_deg, theta3_deg): Joint angles in degrees.
    
    Raises:
        ValueError: If the point is out of reach.
    """
    # 1) XY-plane yaw:
    #    φ = θ1_rad
    phi = math.atan2(y, x)
    #    Map back to user angle:
    theta1_deg = math.degrees(phi + math.pi/2)

    # 2) Project into the shoulder's pitch plane:
    #    Horizontal distance from shoulder:
    r = math.hypot(x, y)
    #    Vertical offset from shoulder:
    dz = z - H

    # 3) Two‑link planar IK:
    D = (r*r + dz*dz - L1*L1 - L2*L2) / (2 * L1 * L2)
    if abs(D) > 1.0:
        raise ValueError(f"Point ({x}, {y}, {z}) is out of reach (|D|={D:.3f}>1).")

    # elbow angle:
    if elbow_up:
        theta3_rad = math.atan2(+math.sqrt(1 - D*D), D)
    else:
        theta3_rad = math.atan2(-math.sqrt(1 - D*D), D)

    # shoulder pitch:
    #    α = atan2(dz, r)
    #    β = atan2(L2*sinθ3, L1 + L2*cosθ3)
    alpha = math.atan2(dz, r)
    beta  = math.atan2(L2 * math.sin(theta3_rad), L1 + L2 * math.cos(theta3_rad))
    theta2_rad = alpha - beta

    # 4) Map back to your original angle conventions:
    #    forward did: θ2_rad = π - rad(theta2_deg)  →  theta2_deg = degrees(π - θ2_rad)
    theta2_deg = math.degrees(math.pi - theta2_rad)

    #    forward did: θ3_rad = rad(theta3_deg) - π  →  rad(theta3_deg) = θ3_rad + π
    theta3_deg = math.degrees(theta3_rad + math.pi)

    return theta1_deg, theta2_deg, theta3_deg

# Connect to the Arduino board on COM3 (change if needed)
board = pyfirmata.Arduino("COM3")

# Start the iterator thread to keep the serial connection alive
it = pyfirmata.util.Iterator(board)
it.start()

# Define the servo pins
servo_pins = [
    board.get_pin('d:4:s'),  # Servo 1 on pin 4
    board.get_pin('d:5:s'),  # Servo 2 on pin 5
    board.get_pin('d:6:s'),  # Servo 3 on pin 6
    #board.get_pin('d:7:s')   # Servo 4 on pin 7, codigo comentado so para testar algo
]

# Give the board a moment to initialize
time.sleep(1)

x = float(input("x?"))
y = float(input("y?"))
z = float(input("z?"))

angles = inverse_kinematics_point(x, y, z, elbow_up=False)


for i, servo in enumerate(servo_pins):
    servo.write(angles[i])
    time.sleep(0.015)


# Wait to allow movement
time.sleep(1)

# Close the board connection
board.sp.close()


