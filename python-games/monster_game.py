import random

class Player:
    def __init__(self, name, health=100):
        self.name = name
        self.health = health

    def attack(self, target):
        damage = random.randint(10, 20)
        target.take_damage(damage)
        print(f"{self.name} attacked {target.name} for {damage} damage.")

    def take_damage(self, damage):
        self.health -= damage
        if self.health <= 0:
            print(f"{self.name} has been defeated!")

class Monster:
    def __init__(self, name, health=50):
        self.name = name
        self.health = health

    def attack(self, target):
        damage = random.randint(10, 20)
        target.take_damage(damage)
        print(f"{self.name} attacked {target.name} for {damage} damage.")

    def take_damage(self, damage):
        self.health -= damage
        if self.health <= 0:
            print(f"{self.name} has been defeated!")

def main():
    player_name = input("Enter your name: ")
    player = Player(player_name)
    monster = Monster("Dragon")

    print("Welcome to the Monster Battle Game!")
    print(f"{player.name} vs {monster.name}")

    while player.health > 0 and monster.health > 0:
        print(f"\n{player.name}: {player.health} health | {monster.name}: {monster.health} health")
        action = input("Do you want to attack the monster? (yes/no): ").lower()

        if action == "yes":
            player.attack(monster)
        elif action == "no":
            print("You chose not to attack.")
        else:
            print("Invalid choice. Please enter 'yes' or 'no'.")

        # Monster's turn
        if monster.health > 0:
            monster.attack(player)

    print("\nGame over!")

if __name__ == "__main__":
    main()
