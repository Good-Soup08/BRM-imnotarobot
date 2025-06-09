import pyfirmata
import time
import os
import numpy as np


#a=np.load("pick_pen.npy") # VIDEO 5
a=np.load("ciao.npy") # VIDEO 3


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
    board.get_pin('d:7:s')   # Servo 4 on pin 7
]

# Give the board a moment to initialize
time.sleep(1)

# Read angles for all 4 servos from user input
for angles in a:
    for i, servo in enumerate(servo_pins):
        servo.write(angles[i])
        time.sleep(0.015)


# Wait to allow movement
time.sleep(1)

# Close the board connection
board.sp.close()
