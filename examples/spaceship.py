import pygame
import math
import numpy as np
import gym
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy
from rl.agents.dqn import DQNAgent
import os
import pathlib

# 宇宙船ゲームの強化学習サンプルコード
# 以下の記事を参考にさせて頂いた
# https://qiita.com/uneyamauneko/items/bcae4f7b64d7188738d7


ROOT_DIR_PATH = str(pathlib.Path(__file__).parent.parent)
MODEL_DIR_PATH = os.path.join(ROOT_DIR_PATH, "models")
MODEl_FILE_PATH = os.path.join(MODEL_DIR_PATH, "spaceship.hdf5")


def make_rnd(a: float, b: float):
    """
    a以上b以下の少数で乱数を生成する
    """
    return np.float(np.random.random_sample() * (b - a) + a)


def shipsin(r, a):
    return math.sin(r + math.pi * 7 / 8 * a) * 5


def shipcos(r, a):
    return math.cos(r + math.pi * 7 / 8 * a) * 5


class game(gym.Env):
    def __init__(self):
        self.screen = pygame.display.set_mode((400, 300))
        self.clock = pygame.time.Clock()
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=np.float32([0, 0, 0]), high=np.float32([400, 300, math.pi]))
        self.reward_range = (-1, 1)
        self.logcount = 9999

    def reset(self):
        self.observation = np.float32([
            make_rnd(0, 100), make_rnd(100, 200), make_rnd(math.pi * 2 / 5, math.pi * 3 / 5)
        ])
        if self.logcount >= 10:
            self.screen.fill((0, 0, 0))
            pygame.draw.circle(self.screen, (100, 255, 255), (300, 150), 20)
            pygame.display.update()
            self.logcount = 0
        self.logcount += 1
        return self.observation

    def render(self, mode):
        x = self.observation[0] + 0.5
        y = self.observation[1] + 0.5
        r = self.observation[2]
        pygame.draw.polygon(self.screen, (255, 255, 255), ((int(x), int(y)), (int(x + shipsin(r, -1)),
                            int(y - shipcos(r, -1))), (int(x + shipsin(r, 1)), int(y - shipcos(r, 1)))))
        pygame.display.update()
        self.clock.tick(60)
        return

    def step(self, act):
        self.observation[2] += np.float(act * 2 - 1) / 20
        self.observation[0] += math.sin(self.observation[2]) * 6
        self.observation[1] -= math.cos(self.observation[2]) * 6
        if self.observation[0] >= 400 or self.observation[1] <= 0 or self.observation[1] >= 300 or self.observation[2] <= 0 or self.observation[2] >= math.pi:
            return self.observation, np.float(-1), True, {}  # 失敗（画面外に出た、船が真上か真下を向いた）、reward=-1
        if (self.observation[0] - 300)**2 + (self.observation[1] - 150)**2 <= 400:
            return self.observation, np.float(1), True, {}  # 成功（船が地球に到着した、地球の半径20)、reward=1
        return self.observation, np.float(0), False, {}  # まだ飛行中、reward=0


pygame.init()
env = game()
env.reset()

model = Sequential([
    Flatten(input_shape=(1, 3)),
    Dense(16, activation='relu'),
    Dense(16, activation='relu'),
    Dense(16, activation='relu'),
    Dense(2, activation='linear')
])
if os.path.exists(MODEl_FILE_PATH):
    model = load_model(MODEl_FILE_PATH)  # 保存したモデルを呼び出す時に使用する
    print(f"Success to load model: {MODEl_FILE_PATH}")
    pass

memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=2, gamma=0.99, memory=memory, nb_steps_warmup=100, target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=1000, visualize=True, verbose=1)  # visualize=Falseにすれば、画面描写をしなくなる

if not os.path.exists(MODEL_DIR_PATH):
    os.makedirs(MODEL_DIR_PATH, exist_ok=True)

dqn.model.save(MODEl_FILE_PATH, overwrite=True)
dqn.test(env, nb_episodes=10, visualize=True)
pygame.quit()
