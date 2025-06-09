import numpy as np
import random
from app import Gestures

class TicTacToe:
    all_numbers = list(range(0, 11))
    random.shuffle(all_numbers)
    selected_numbers = all_numbers[:9]
    
    def __init__(self):
        self.board = np.zeros((3,3), dtype=int)
        self.available = 9

    def write(self, row, column, symbol):
        self.board[row][column] = symbol

    def spaceUsed(self, row, column):
        if (self.board[row][column] != 0):
            return True
        else:
            return False

    def play(self, row, column, symbol):
        if (self.spaceUsed(row, column)):
            return False
        else:
            self.write(row, column, symbol)
            self.available -= 1
            return True
        
    def gameEnd(self):
        diag_sums = np.array([np.trace(self.board), np.trace(np.fliplr(self.board))])
        sums = np.hstack((np.sum(self.board, axis=0), np.sum(self.board, axis=1), diag_sums))

        if (np.any(sums == 3)): # X wins
            return 1
        elif (np.any(sums == -3)): # O wins
            return -1
        else:
            if (self.available == 0): # no more moves, draw
                return 0
            else: # continue playing
                return 2
    
    def show(self): # for debugging reasons only
        for row in self.board:
            for element in row:
                print(element, end="|")
            print("\n")

    def main(self, draw_grid): 
        ml = Gestures()
        symbol = 1
        while (True):
            draw_grid(self.selected_numbers)
            self.show()
            
            inp = 0
            while (not (int(inp) >= 1 and int(inp) <= 9)):
                ml.main()
                gesture_number = int(ml.get_gesture3())
                for index in range(np.size(self.selected_numbers)):
                    if (self.selected_numbers[index] == gesture_number):
                        inp = index + 1
                        break
                if (gesture_number == -1):  # -1 when there is no gesture collected
                    print("Try again with a gesture in the time limit of the camera.")
                    draw_grid(self.selected_numbers)
                    self.show()
                elif (inp == 0):
                    print("Invalid Gesture, try one of the ones presented in the grid.")
                    draw_grid(self.selected_numbers)
                    self.show()
                    
            column = (int(inp) % 3) - 1
            row = ((int(inp)-1) // 3)
            
            success = self.play(row, column, symbol)
            if not success:
                print("Place already used!") 
                symbol = -symbol

            symbol = -symbol

            outcome = self.gameEnd()
            if outcome == 0:
                print('Draw')
                self.show()
                break
            elif outcome == 1:
                print('X wins')
                self.show()
                break
            elif outcome == 2:
                print("\n\n")
            elif outcome == -1:
                print('O wins')
                self.show()
                break
                
            print("Remaining: ", self.available, "\n")
