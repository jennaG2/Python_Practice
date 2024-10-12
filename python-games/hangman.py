import random

class Hangman:
    def __init__(self, word_list):
        self.word_to_guess = random.choice(word_list).upper()
        self.guesses_left = 6
        self.correct_guesses = set()
        self.incorrect_guesses = set()

    def display_word(self):
        display = ""
        for letter in self.word_to_guess:
            if letter in self.correct_guesses:
                display += letter + " "
            else:
                display += "_ "
        return display.strip()

    def make_guess(self, guess):
        guess = guess.upper()

        if guess.isalpha() and len(guess) == 1:
            if guess in self.correct_guesses or guess in self.incorrect_guesses:
                print("You already guessed that letter.")
            elif guess in self.word_to_guess:
                print("Correct guess!")
                self.correct_guesses.add(guess)
            else:
                print("Incorrect guess.")
                self.incorrect_guesses.add(guess)
                self.guesses_left -= 1
        else:
            print("Invalid guess. Please enter a single alphabetical character.")

    def is_game_over(self):
        return self.guesses_left == 0 or set(self.correct_guesses) == set(self.word_to_guess)

    def display_result(self):
        if set(self.correct_guesses) == set(self.word_to_guess):
            print(f"Congratulations! You guessed the word: {self.word_to_guess}")
        else:
            print(f"Sorry, you're out of guesses. The word was: {self.word_to_guess}")

def main():
    word_list = ["python", "hangman", "programming", "computer", "algorithm", "developer"]
    hangman_game = Hangman(word_list)

    print("Welcome to Hangman!")
    print(hangman_game.display_word())

    while not hangman_game.is_game_over():
        guess = input("Enter your guess: ")
        hangman_game.make_guess(guess)
        print("Word:", hangman_game.display_word())
        print(f"Guesses left: {hangman_game.guesses_left}\n")

    hangman_game.display_result()

if __name__ == "__main__":
    main()
