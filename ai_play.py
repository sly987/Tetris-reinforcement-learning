import pygame
import pickle
from matris_core import MatrisCore

BLOCK = 30
WIDTH = 10 * BLOCK
HEIGHT = 22 * BLOCK

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
font = pygame.font.Font(None, 36)

with open("qtable.pkl", "rb") as f:
    q_table = pickle.load(f)

env = MatrisCore()
clock = pygame.time.Clock()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    obs = env.get_observation()
    state = env.state_to_key(obs)
    action = q_table[state].argmax() if state in q_table else 3

    _, reward, done = env.step(action)

    if done:
        print("GAME OVER — Reward total :", env.total_reward)
        pygame.time.wait(1500)
        env.reset()
        continue

    screen.fill((15, 15, 20))

    for y in range(env.matrix.shape[0]):
        for x in range(env.matrix.shape[1]):
            if env.matrix[y][x]:
                pygame.draw.rect(
                    screen,
                    (100, 100, 255),
                    (x * BLOCK, y * BLOCK, BLOCK, BLOCK)
                )

    reward_text = font.render(
        f"Reward score : {int(env.total_reward)}",
        True,
        (255, 255, 255)
    )
    screen.blit(reward_text, (20, 20))

    pygame.display.flip()
    clock.tick(15)

