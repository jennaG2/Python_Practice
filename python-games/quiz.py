import random

class Question:
    def __init__(self, text, options, correct_option):
        self.text = text
        self.options = options
        self.correct_option = correct_option

    def is_correct(self, user_answer):
        return user_answer == self.correct_option

class QuizGame:
    def __init__(self, questions):
        self.questions = questions
        self.score = 0

    def display_question(self, question):
        print(question.text)
        for i, option in enumerate(question.options, start=1):
            print(f"{i}. {option}")
        user_answer = input("Enter the number of your answer: ")
        return int(user_answer)

    def play(self):
        random.shuffle(self.questions)
        for question in self.questions:
            user_answer = self.display_question(question)
            if question.is_correct(user_answer):
                print("Correct! You earned 10 points.")
                self.score += 10
            else:
                print(f"Wrong! The correct answer was {question.correct_option}.")
            print(f"Your current score: {self.score}\n")

        print("Quiz completed!")
        print(f"Your final score: {self.score}")

# Sample questions
questions = [
    Question("What is the capital of France?", ["Berlin", "Madrid", "Paris", "Rome"], 3),
    Question("Which programming language is known for its readability?", ["Java", "C++", "Python", "JavaScript"], 3),
    Question("What is the largest planet in our solar system?", ["Mars", "Jupiter", "Venus", "Saturn"], 2),
    # Add more questions as needed
]

# Create a quiz game and play it
if __name__ == "__main__":
    quiz_game = QuizGame(questions)
    quiz_game.play()
