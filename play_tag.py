import pygame
from envs.tag import TagSimpleGame, TagPlayer
import typing as t


class TagPlayerHuman(TagPlayer):
    def __init__(self, x: int, y: int, radius: int = 5, velocity: int = 0, color: t.Tuple[int, int, int] = (255, 0, 0)) -> None:
        super().__init__(x, y, radius, velocity, color)
        return

    def get_action(self):
        pressed_key = pygame.key.get_pressed()

        action = TagSimpleGame.ACTION_STAY

        if pressed_key[pygame.K_LEFT]:
            action = TagSimpleGame.ACTION_LEFT
        elif pressed_key[pygame.K_UP]:
            action = TagSimpleGame.ACTION_UP
        elif pressed_key[pygame.K_RIGHT]:
            action = TagSimpleGame.ACTION_RIGHT
        elif pressed_key[pygame.K_DOWN]:
            action = TagSimpleGame.ACTION_DOWN

        return action


player = TagPlayerHuman(x=0, y=0, radius=TagSimpleGame.RADIUS, velocity=TagSimpleGame.VELOCITY)

env = TagSimpleGame(demon=player)

episodes = 20

for episode in range(episodes):
    env.reset()
    is_done = False
    score = 0

    while not is_done:
        env.render(None)
        action = player.get_action()
        observations, reward, is_done, _ = env.step(action=action)
        score += reward

        continue

    print(f"episode: {episode}, score: {score}")

    continue

env.close()
