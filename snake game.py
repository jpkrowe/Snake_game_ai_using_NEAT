import pygame
import neat
import os
import random
import math
pygame.font.init()


# define screen size
WIN_WIDTH = 500
WIN_HEIGHT = 500
SNAKE_WIDTH = 10
SNAKE_INIT_X = 150
SNAKE_INIT_Y = 150

GEN = 0

STAT_FONT = pygame.font.SysFont("comicsans", 50)



class Head:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.rot_clockwise = math.pi/2
        self.rot_anticlockwise = -math.pi/2
        self.vel = SNAKE_WIDTH
        self.angle = 0
        self.previous_x = 0
        self.previous_y = 0


    def move(self):
        self.previous_x = self.x
        self.previous_y = self.y
        self.x = self.x + self.vel*math.cos(self.angle)
        self.y = self.y + self.vel*math.sin(self.angle)

    def rotate(self, rotate):
        if rotate == 1:
            self.angle = self.angle + self.rot_clockwise

        elif rotate == 2:
            self.angle = self.angle + self.rot_anticlockwise

    def draw(self, win):
        pygame.draw.rect(win, (255, 0, 0), (self.x, self.y, SNAKE_WIDTH, SNAKE_WIDTH))


class Body:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.previous_x = x
        self.previous_y = y

    def move(self, head):
        self.x = head.previous_x
        self.y = head.previous_y

    def draw(self, win):
        pygame.draw.rect(win, (255, 0, 0), (self.x, self.y, SNAKE_WIDTH, SNAKE_WIDTH))


class Food:
    def __init__(self):
        self.x = random.randint(0, WIN_WIDTH)
        self.y = random.randint(0, WIN_HEIGHT)

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
        for i in range(1, 10):
            snake.append(Body(SNAKE_INIT_X - i*SNAKE_WIDTH, SNAKE_INIT_Y))


    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()
    pygame.display.set_caption("Snake")

    for i, snake in enumerate(snakes):
        food = Food()
        alive = True
        score = 0
        while alive:
            clock.tick(10)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

            # finds the nearest wall
            if snake[0].x < WIN_WIDTH/2:
                xwall = snake[0].x
            else:
                xwall = WIN_WIDTH - snake[0].x
            if snake[0].y < WIN_HEIGHT/2:
                ywall = snake[0].y
            else:
                ywall = WIN_HEIGHT - snake[0].y
            if xwall < ywall:
                nearestWall = xwall
            else:
                nearestWall = ywall

            # inputs the data into the neural network
            output = nets[i].activate((food.x - snake[0].x, food.y - snake[0].y, abs(nearestWall)))
            if output[0] > 0.5:
                snake[0].rotate(2)
            elif output[1] > 0.5:
                snake[0].rotate(1)

            # moves the snake
            snake[0].move()
            snake[-1].move(snake[0])
            snake.insert(1, snake[-1])
            snake.pop(-1)
            win.fill((0, 0, 0))
            draw_window(win, snake, food, score)
            pygame.display.update()

            if food.x == snake[0].x and food.y == snake[0].y:
                snake.append(Body(snake[-1].x, snake[-1] + SNAKE_WIDTH))
                score += 1
                ge[i].fitness += 10
                food = Food()  # generates new food

            # checks if the snake hits itself
            for body in snake[1:]:
                if (snake[0].x - body.x) < SNAKE_WIDTH and (snake[0].y - body.y) < SNAKE_WIDTH:
                    alive = False
                    ge[i].fitness -= 1

            # checks if the snake has hit a wall
            if snake[0].x <= 0 or snake[0].x >= WIN_WIDTH or snake[0].y <= 0 or snake[0].y >= WIN_HEIGHT:
                alive = False
                ge[i].fitness -= 1

            # gives fitness for staying alive
            if alive:
                ge[i].fitness += 1

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