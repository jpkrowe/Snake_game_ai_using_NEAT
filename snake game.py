import pygame
import neat
import os
import random
pygame.font.init()


# define screen size
GRID_X = 30
GRID_Y = 30
SNAKE_WIDTH = 10
WIN_WIDTH = GRID_X * SNAKE_WIDTH
WIN_HEIGHT = GRID_Y * SNAKE_WIDTH
SNAKE_INIT_X = 10
SNAKE_INIT_Y = 10

INIT_SNAKE_LENGTH = 6

GEN = 0

STAT_FONT = pygame.font.SysFont("comicsans", 50)



class Head:

    def __init__(self, xcoord, ycoord):
        self.xcoord = xcoord
        self.ycoord = ycoord
        self.x = self.xcoord * SNAKE_WIDTH
        self.y = self.ycoord * SNAKE_WIDTH
        self.travelDirection = [1, 0]
        self.previous_x = 0
        self.previous_y = 0
        self.rect = pygame.Rect(self.x, self.y, SNAKE_WIDTH, SNAKE_WIDTH)
        self.previous_xcoord = 0
        self.previous_ycoord = 0

    def move(self):
        self.previous_xcoord = self.xcoord
        self.previous_ycoord = self.ycoord

        self.xcoord = self.xcoord + self.travelDirection[0]
        self.ycoord = self.ycoord + self.travelDirection[1]
        self.x = self.xcoord * SNAKE_WIDTH
        self.y = self.ycoord * SNAKE_WIDTH
        self.rect = pygame.Rect(self.x, self.y, SNAKE_WIDTH, SNAKE_WIDTH)

    def rotate(self, rotate):

        if rotate == 1: # clockwise turn
            if self.travelDirection[1] == 0:
                placeholder = self.travelDirection[0]
                self.travelDirection[0] = self.travelDirection[1]
                self.travelDirection[1] = -placeholder
            else:
                placeholder = self.travelDirection[0]
                self.travelDirection[0] = self.travelDirection[1]
                self.travelDirection[1] = placeholder
        elif rotate == 2:  # anticlockwise turn
            if self.travelDirection[0] == 0:
                placeholder = self.travelDirection[0]
                self.travelDirection[0] = -self.travelDirection[1]
                self.travelDirection[1] = placeholder
            else:
                placeholder = self.travelDirection[0]
                self.travelDirection[0] = self.travelDirection[1]
                self.travelDirection[1] = placeholder

    def draw(self, win):
        pygame.draw.rect(win, (255, 0, 0), (self.x, self.y, SNAKE_WIDTH, SNAKE_WIDTH))

    def sense(self, rotoption):  # generates the next potential position of the head piece
        testDirection = self.travelDirection
        if rotoption == 1:  # clockwise turn
            if testDirection[1] == 0:
                placeholder = testDirection[0]
                testDirection[0] = testDirection[1]
                testDirection[1] = -placeholder
            else:
                placeholder = testDirection[0]
                testDirection[0] = testDirection[1]
                testDirection[1] = placeholder
        elif rotoption == 2:  # anticlockwise turn
            if testDirection[0] == 0:
                placeholder = testDirection[0]
                testDirection[0] = -testDirection[1]
                testDirection[1] = placeholder
            else:
                placeholder = testDirection[0]
                testDirection[0] = testDirection[1]
                testDirection[1] = placeholder

        testcoord = [self.xcoord + testDirection[0], self.ycoord + testDirection[1]]
        return testcoord



class Body:
    def __init__(self, xcoord, ycoord):
        self.xcoord = xcoord
        self.ycoord = ycoord
        self.x = self.xcoord * SNAKE_WIDTH
        self.y = self.ycoord * SNAKE_WIDTH
        self.rect = pygame.Rect(self.x, self.y, SNAKE_WIDTH, SNAKE_WIDTH)

    def move(self, head):
        self.xcoord = head.previous_xcoord
        self.ycoord = head.previous_ycoord
        self.x = self.xcoord*SNAKE_WIDTH
        self.y = self.ycoord*SNAKE_WIDTH
        self.rect = pygame.Rect(self.x, self.y, SNAKE_WIDTH, SNAKE_WIDTH)

    def draw(self, win):
        pygame.draw.rect(win, (255, 0, 0), (self.x, self.y, SNAKE_WIDTH, SNAKE_WIDTH))


class Food:
    def __init__(self):
        self.xcoord = random.randint(0, GRID_X-1)
        self.ycoord = random.randint(0, GRID_Y-1)
        self.x = self.xcoord*SNAKE_WIDTH
        self.y = self.ycoord*SNAKE_WIDTH
        self.rect = pygame.Rect(self.x, self.y, SNAKE_WIDTH, SNAKE_WIDTH)

    def draw(self, win):
        pygame.draw.rect(win, (255, 0, 0), (self.x, self.y, SNAKE_WIDTH, SNAKE_WIDTH))


def draw_window(win, snake, food, score):

    for body_part in snake:
        body_part.draw(win)

    food.draw(win)

    text = STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
    win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))

    text = STAT_FONT.render("Gen: " + str(GEN), 1, (255, 255, 255))
    win.blit(text, (10, 10))


def main(genomes, config):
    global GEN
    GEN += 1
    nets = []
    ge = []
    snakes = []

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        snake = [Head(SNAKE_INIT_X, SNAKE_INIT_Y)]
        snakes.append(snake)
        g.fitness = 0
        ge.append(g)
    for snake in snakes:
        for i in range(1, INIT_SNAKE_LENGTH + 1):
            snake.append(Body(SNAKE_INIT_X - i, SNAKE_INIT_Y))


    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()
    pygame.display.set_caption("Snake")

    for i, snake in enumerate(snakes):
        food = Food()
        alive = True
        score = 0
        noRotClock = 0
        noRotAntiClock = 0
        missFood = 0


        while alive:
            clock.tick(500)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        snake[0].rotate(1)
                    if event.key == pygame.K_RIGHT:
                        snake[0].rotate(2)

            # inputs the data into the neural network
            inputs = [0, 0, 0]
            for rotoption in range(0, 3):  # rotoption = 0 straight  rotoption = 1 clockwise turn  rotoption = 2 anticlockwise turn
                collisionTest = snake[0].sense(rotoption)
                # checks if the snake will hit itself
                for body in snake[1:]:
                    if collisionTest[0] == body.xcoord and collisionTest[1] == body.ycoord:
                        inputs[rotoption] = 1

                # checks if the snake will hit a wall
                if collisionTest[0] < 0 or collisionTest[1] < 0 or collisionTest[0] >= GRID_X or collisionTest[1] >= GRID_Y:
                    inputs[rotoption] = 1

            foodDistance = (snake[0].xcoord - food.xcoord)**2 + (snake[0].ycoord - food.ycoord)**2

            output = nets[i].activate((foodDistance, inputs[0], inputs[1], inputs[2]))
            if output[0] > 0.5 and output[1] > 0.5:
                pass
            elif output[0] > 0.5:

                snake[0].rotate(1)

            elif output[1] > 0.5:
                snake[0].rotate(2)

            # moves the snake
            snake[0].move()
            snake[-1].move(snake[0])
            snake.insert(1, snake[-1])
            snake.pop(-1)
            win.fill((0, 0, 0))
            draw_window(win, snake, food, score)
            pygame.display.update()

            if snake[0].xcoord == food.xcoord and snake[0].ycoord == food.ycoord:
                snake.append(Body(-1, -1))
                score += 1
                ge[i].fitness += 75
                food = Food()  # generates new food
                missFood = 0
            else:
                missFood += 1


            # checks if the snake has hit itself
            for body in snake[1:]:
                if snake[0].xcoord == body.xcoord and snake[0].ycoord == body.ycoord:
                    alive = False
                    ge[i].fitness -= 15

            # checks if the snake has hit a wall
            if snake[0].xcoord < 0 or snake[0].ycoord < 0 or snake[0].xcoord >= GRID_X or snake[0].ycoord >= GRID_Y:
                alive = False
                ge[i].fitness -= 15

            # Kills the snake if it has gone too long without getting food
            if missFood > 750:
                ge[i].fitness -= 30
                alive = False

            # gives fitness for staying alive
            if alive:
                ge[i].fitness += 0.1

            if not alive:
                snakes.pop(i)
                nets.pop(i)
                ge.pop(i)


def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(main, 50)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)