import random
import time

def show_rules():
    print("🎮 Tic-Tac-Toe Rules:")
    print("- Two players take turns.")
    print("- The goal is to get three symbols (X or O) in a row, column, or diagonal.")
    print("- In this version, the robotic arm is the one who plays, and it receives indication to where to move from the player's chosen gesture.")
    print("- You have indications of a 3x3 grid with a number for each square. Those numbers are randomly chosen from 1 to 12.")
    print("- Each number is associated with a gesture indicated by the image.")
    print("- The player needs to do a hand gesture to the camera to indicate the number of the square where the robot should move.")
    print("- The player has 10 seconds to play.")
    print("- Players will take turns, so it's not possible for them to do a hand gesture at the same time.")
    print("- If you don't play, the robot will decide a random place.") # isto n faz sentido, mais vale deixar fzr input keyboard
    print("- The game ends when all the squares are occupied or if someone has won. Good luck! \n")
    # dizer que vai ter x tempo para jogar

def countdown():
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    print("GO!\n")

def draw_grid(numbers):
    print("Game Grid (choose based on the gesture corresponding to the number):\n")
    print("    📍🤖")
    print("    robot")
    print("     is")
    print("    here\n")
    for i in range(3):
        row = " | ".join(f"{numbers[i*3 + j]:>2}" for j in range(3))
        print(row)
        if i < 2:
            print("---+----+---")

# Main Program
show_rules()

from ttt_engine import TicTacToe
# Play the game
ttt = TicTacToe()

# Choose 9 unique numbers out of 12
# selected_numbers = ttt.selected_numbers

# Draw the 3x3 game grid
# draw_grid(selected_numbers)

countdown()

# nova linha
ttt.main(draw_grid)
