import serial
import time
import numpy as np
import os

arduino = serial.Serial('COM3', 9600)  # Change COM3 to your Arduino port
time.sleep(2)  # wait for the connection to initialize

l0 = 6.8
l1 = 6.1
l2 = 8

def ik(x,y,z):
    global l0, l1, l2

    theta0 = np.arctan2(y,x)   # returns [-π, π]

    dist = np.sqrt(x**2 + y**2 + (z-l0)**2)
    arg1 = (dist**2 - l1**2 - l2**2)/(2*l1*l2)
    if np.abs(arg1) > 1:
        arg1 = np.sign(arg1)
    theta2 = np.arccos(arg1)

    beta = np.arccos((z-l0)/dist)
    arg2 = (l2**2 - l1**2 - dist**2)/(-2*l1*dist)
    if np.abs(arg2) > 1:
        arg2 = np.sign(arg2)
    alpha = np.arccos(arg2)
    theta1 = beta - alpha
    
    # transformar para angulos do robot:
    theta0 = theta0 + np.pi/2
    theta1 = theta1 + np.pi/2
    theta2 = -theta2 + np.pi

    theta0 = theta0 % (2 * np.pi)
    theta1 = theta1 % (2 * np.pi)
    theta2 = theta2 % (2 * np.pi)

    return theta0, theta1, theta2

def set_servo(servo_number, angle):
    if 0 <= servo_number <= 3 and 0 <= angle <= 360:
        command = f"{servo_number}:{angle}\n"
        arduino.write(command.encode())
        time.sleep(1.5)  # Give Arduino time to respond

def move_arm(j0, j1, j2, j3):

    set_servo(3,j3)
    set_servo(2,j2)
    set_servo(1,j1)
    set_servo(0,j0)

    time.sleep(0.1)


set_servo(2,80)
"""
x = np.array(range(5,15)).reshape(10, 1)
y = np.full((10,1), 4.5)
z = np.full((10,1), 8)
arr = np.vstack([x, y, z])

for i in range(10):
    angles = ik(x[i],y[i],z[i])
    angles = np.degrees(angles)

    move_arm(angles[0],angles[1],angles[2],165)
    print(angles)
"""






#for row in arr:
#    move_arm(row[0],row[1],row[2],row[3])





