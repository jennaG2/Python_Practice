import random

class BattleshipGame:
    def __init__(self):
        self.board_size = 5
        self.board = [['O' for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.ships = 3
        self.remaining_ships = self.ships
        self.target_row = 0
        self.target_col = 0

    def print_board(self):
        for row in self.board:
            print(" ".join(row))
        print()

    def place_ships(self):
        for _ in range(self.ships):
            ship_row = random.randint(0, self.board_size - 1)
            ship_col = random.randint(0, self.board_size - 1)
            self.board[ship_row][ship_col] = 'S'

    def get_user_guess(self):
        while True:
            try:
                guess_row = int(input("Guess Row (0 to {}): ".format(self.board_size - 1)))
                guess_col = int(input("Guess Col (0 to {}): ".format(self.board_size - 1)))
                if 0 <= guess_row < self.board_size and 0 <= guess_col < self.board_size:
                    return guess_row, guess_col
                else:
                    print("Please enter valid coordinates.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    def play_turn(self):
        print("Take a shot!")
        guess_row, guess_col = self.get_user_guess()

        if self.board[guess_row][guess_col] == 'S':
            print("Congratulations! You sunk a battleship!")
            self.remaining_ships -= 1
        elif self.board[guess_row][guess_col] == 'X':
            print("You've already guessed that one!")
        else:
            print("You missed!")
            self.board[guess_row][guess_col] = 'X'

    def computer_turn(self):
        print("\nComputer's turn!")
        self.target_row = random.randint(0, self.board_size - 1)
        self.target_col = random.randint(0, self.board_size - 1)

        if self.board[self.target_row][self.target_col] == 'S':
            print("The computer sunk your battleship!")
            self.remaining_ships -= 1
        else:
            print("The computer missed!")
            self.board[self.target_row][self.target_col] = 'X'

    def play_game(self):
        print("Welcome to Battleship!")
        self.place_ships()

        while self.remaining_ships > 0:
            self.print_board()
            self.play_turn()

            if self.remaining_ships > 0:
                self.computer_turn()

        print("Game over! You {} the game.".format("won" if self.remaining_ships == 0 else "lost"))

# Start the game
game = BattleshipGame()
game.play_game()
