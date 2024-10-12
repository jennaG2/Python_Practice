import pygame, sys

pygame.init()
#adding Green color
GREEN = (173, 204, 96)
DARK_GREEN = (43,51,24)

screen = pygame.display.set_mode((750,750))

pygame.display.set_caption("Jenna's Snake Game")

clock = pygame.time.Clock()

#Game Loop

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    screen.fill(GREEN)
    pygame.display.update()
    clock.tick(60)