class ConnectFour:
    def __init__(self):
        self.board = [[' ' for _ in range(7)] for _ in range(6)]
        self.current_player = 'X'

    def print_board(self):
        for row in self.board:
            print("|".join(row))
            print("-" * 29)

    def is_column_full(self, col):
        return self.board[0][col] != ' '

    def drop_piece(self, col):
        for row in range(5, -1, -1):
            if self.board[row][col] == ' ':
                self.board[row][col] = self.current_player
                break
        else:
            print("Column is full. Choose another column.")

    def check_winner(self, row, col):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1
            for i in range(1, 4):
                r, c = row + i * dr, col + i * dc
                if 0 <= r < 6 and 0 <= c < 7 and self.board[r][c] == self.current_player:
                    count += 1
                else:
                    break

            for i in range(1, 4):
                r, c = row - i * dr, col - i * dc
                if 0 <= r < 6 and 0 <= c < 7 and self.board[r][c] == self.current_player:
                    count += 1
                else:
                    break

            if count >= 4:
                return True

        return False

    def switch_player(self):
        self.current_player = 'O' if self.current_player == 'X' else 'X'

    def play_game(self):
        print("Welcome to Connect Four!")
        while True:
            self.print_board()

            try:
                col = int(input(f"Player {self.current_player}, choose a column (0-6): "))
                if 0 <= col <= 6 and not self.is_column_full(col):
                    self.drop_piece(col)
                    if self.check_winner(5 - self.board[0].count(' '), col):
                        self.print_board()
                        print(f"Player {self.current_player} wins!")
                        break
                    self.switch_player()
                else:
                    print("Invalid column choice. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")

# Start the game
connect_four = ConnectFour()
connect_four.play_game()
