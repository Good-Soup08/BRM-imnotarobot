import pyfirmata
import time
import os
import numpy as np

#a=np.load("paths/pick_pen.npy") # VIDEO for teaser
a=np.load("paths/ciao.npy") # VIDEO for teaser

# connecting to the Arduino board on COM3
board = pyfirmata.Arduino("COM3")

# start an iterator thread to use with the pins
it = pyfirmata.util.Iterator(board)
it.start()

# define the servo pins
servo_pins = [
    board.get_pin('d:4:s'),  # Servo 1 on pin 4, "s" means servo mode
    board.get_pin('d:5:s'),  # Servo 2 on pin 5
    board.get_pin('d:6:s'),  # Servo 3 on pin 6
    board.get_pin('d:7:s')   # Servo 4 on pin 7
]

# give the board a moment to initialize
time.sleep(1)

for angles in a:
    for i, servo in enumerate(servo_pins):
        servo.write(angles[i])
        time.sleep(0.015) # dead time between commands

# time to finnish executing the movement
time.sleep(1)

# end the connection
board.sp.close()
