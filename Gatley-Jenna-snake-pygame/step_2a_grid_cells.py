import pygame, sys

pygame.init()
#adding Green color
GREEN = (173, 204, 96)
DARK_GREEN = (43,51,24)
#adding cells size and number of cells
cell_size = 30
number_of_cells = 25

#screen = pygame.display.set_mode((750,750))
screen = pygame.display.set_mode((cell_size * number_of_cells, cell_size * number_of_cells))

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