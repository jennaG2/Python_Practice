import random

class OregonTrailGame:
    def __init__(self):
        self.player_name = ""
        self.current_mileage = 0
        self.food = 100
        self.health = 100
        self.money = 200
        self.days_on_trail = 0
        self.miles_to_go = 2000

    def start_game(self):
        print("Welcome to the Oregon Trail!")
        self.player_name = input("Enter your name: ")
        print(f"Hello, {self.player_name}! Your journey begins.")

        while self.miles_to_go > 0 and self.health > 0:
            self.print_status()
            self.user_choice()

        if self.miles_to_go <= 0:
            print("Congratulations! You have reached your destination!")
        else:
            print("Game over. You did not make it to your destination.")

    def print_status(self):
        print("\n----- Status -----")
        print(f"Name: {self.player_name}")
        print(f"Days on the trail: {self.days_on_trail}")
        print(f"Miles to go: {self.miles_to_go}")
        print(f"Food: {self.food}")
        print(f"Health: {self.health}")
        print(f"Money: ${self.money}")
        print("------------------\n")

    def user_choice(self):
        print("What will you do?")
        print("1. Continue on the trail")
        print("2. Hunt for food")
        print("3. Rest")
        choice = input("Enter your choice (1/2/3): ")

        if choice == "1":
            self.continue_on_trail()
        elif choice == "2":
            self.hunt_for_food()
        elif choice == "3":
            self.rest()
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

    def continue_on_trail(self):
        distance_covered = random.randint(50, 100)
        self.current_mileage += distance_covered
        self.miles_to_go -= distance_covered
        self.days_on_trail += 1
        self.food -= 10
        self.health -= 5

    def hunt_for_food(self):
        food_found = random.randint(10, 30)
        self.food += food_found
        self.days_on_trail += 1
        self.health -= 5
        print(f"You found {food_found} pounds of food while hunting.")

    def rest(self):
        days_to_rest = random.randint(1, 3)
        self.days_on_trail += days_to_rest
        self.health += 10
        print(f"You rested for {days_to_rest} days and regained some health.")

# Create an instance of the game and start it
game = OregonTrailGame()
game.start_game()
