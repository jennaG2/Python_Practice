import random

class NumberGuessingGame:
    def __init__(self):
        self.secret_number = random.randint(1, 100)
        self.attempts = 0
        self.max_attempts = 10

    def get_user_guess(self):
        while True:
            try:
                guess = int(input("Enter your guess (1-100): "))
                if 1 <= guess <= 100:
                    return guess
                else:
                    print("Please enter a number between 1 and 100.")
            except ValueError:
                print("Please enter a valid number.")

    def play(self):
        print("Welcome to the Number Guessing Game!")
        print("Try to guess the secret number between 1 and 100.")

        while self.attempts < self.max_attempts:
            guess = self.get_user_guess()
            self.attempts += 1

            if guess < self.secret_number:
                print("Too low! Try again.")
            elif guess > self.secret_number:
                print("Too high! Try again.")
            else:
                print(f"Congratulations! You guessed the number {self.secret_number} in {self.attempts} attempts.")
                break

        if self.attempts == self.max_attempts:
            print(f"Sorry, you've run out of attempts. The secret number was {self.secret_number}.")

if __name__ == "__main__":
    game = NumberGuessingGame()
    game.play()
