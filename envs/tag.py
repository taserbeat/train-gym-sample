import gym
from gym import spaces
import pygame
import numpy as np
import random
import typing as t


class TagSimpleGame(gym.Env):
    """
    簡単な鬼ごっこをするゲーム環境

    環境:
        - 赤: 鬼役(主人公)
        - 青: 逃走役(ただし、簡単化のため固定)
        - 鬼役、逃走役の開始位置はどちらもランダム
        - 鬼が一定時間以内に逃走役を捕まえなければゲーム終了

    行動:
        鬼は5種類の行動を取る

        上下左右の4方向への移動、またはその場に留まる

    観測:
        鬼は逃走役までの相対座標を観測できる

        (x_rel, y_rel)

    報酬:
        TODO: 追記
    """

    # 共通パラメータ
    FIELD_X_MIN, FILED_X_MAX = 0, 400  # フィールドのx座標範囲
    FIELD_Y_MIN, FIELD_Y_MAX = 100, 400  # フィールドのy座標範囲
    FPS = 60  # ゲームのフレームレート
    TIME_LIMIT_SEC = 10  # 1ゲームあたりの制限時間 [単位: 秒]

    RADIUS = 5  # 鬼ごっこプレイヤー(円)の半径
    VELOCITY = 5  # 移動速度 [単位: ピクセル/フレーム]

    # 行動の種類
    ACTION_STAY = 0
    ACTION_LEFT = 1
    ACTION_UP = 2
    ACTION_RIGHT = 3
    ACTION_DOWN = 4

    def __init__(self) -> None:
        # 学習の設定
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=np.float32([0, 0]), high=np.float32([400, 300]))  # type: ignore
        self.reward_range = (-1, 1)

        # ゲーム設定
        pygame.init()
        self.screen = pygame.display.set_mode((self.FILED_X_MAX, self.FILED_X_MAX))
        pygame.display.set_caption("鬼ごっこの強化学習")
        self.clock = pygame.time.Clock()

        # ゲームの変数
        self.limit_frame = self.FPS * self.TIME_LIMIT_SEC
        self.remain_frame = self.limit_frame
        self.demon = TagPlayer(x=0, y=0, radius=self.RADIUS, velocity=self.VELOCITY, color=(255, 0, 0))
        self.fugitive = TagPlayer(x=0, y=0, radius=self.RADIUS, color=(0, 0, 255))

        return

    def reset(self):
        self.remain_frame = self.limit_frame

        self.demon.set_randomize_position(
            x_min=0 + self.demon.radius,
            x_max=self.FILED_X_MAX - self.demon.radius,
            y_min=300 + self.demon.radius,
            y_max=self.FIELD_Y_MAX - self.demon.radius
        )

        self.fugitive.set_randomize_position(
            x_min=0 + self.fugitive.radius,
            x_max=self.FILED_X_MAX - self.fugitive.radius,
            y_min=self.FIELD_Y_MIN + self.fugitive.radius,
            y_max=200 - self.fugitive.radius
        )

        self.screen.fill((0, 0, 0))

        return self._calc_observations()

    def step(self, action):
        self.remain_frame -= 1

        # 座標の更新
        if action == self.ACTION_STAY:
            pass
        if action == self.ACTION_LEFT:
            self.demon.x -= self.demon.velocity
        if action == self.ACTION_UP:
            self.demon.y -= self.demon.velocity
        if action == self.ACTION_RIGHT:
            self.demon.x += self.demon.velocity
        if action == self.ACTION_DOWN:
            self.demon.y += self.demon.velocity

        # 座標の修正
        if self.demon.x - self.demon.radius < self.FIELD_X_MIN:
            self.demon.x = self.FIELD_X_MIN + self.demon.radius
        if self.demon.x + self.demon.radius > self.FILED_X_MAX:
            self.demon.x = self.FILED_X_MAX - self.demon.radius
        if self.demon.y - self.demon.radius < self.FIELD_Y_MIN:
            self.demon.y = self.FIELD_Y_MIN + self.demon.radius
        if self.demon.y + self.demon.radius > self.FIELD_Y_MAX:
            self.demon.y = self.FIELD_Y_MAX - self.demon.radius

        # 判定処理のための計算
        # 鬼を基準とした逃走者の相対座標、鬼と逃走者の距離
        position_2d_rel = self.demon.calc_rel_position_2d(self.fugitive.position_2d)
        distance = np.linalg.norm(position_2d_rel)

        # 鬼が捕まえた場合
        if self.demon.radius + self.fugitive.radius > distance:
            # (観測値, 報酬, True, {})を返す
            reward = np.float(self.remain_frame / self.limit_frame)
            return self._calc_observations(), reward, True, {}

        # タイムオーバー
        if self.remain_frame <= 0:
            # (観測値, 報酬, True, {})を返す処理
            reward = np.float(-1)
            return self._calc_observations(), reward, True, {}

        # ゲーム続行中
        return self._calc_observations(), np.float(0), False, {}

    def render(self, mode):
        self.clock.tick(self.FPS)
        self.screen.fill((0, 0, 0))

        pygame.draw.circle(self.screen, self.demon.color, self.demon.position_2d, self.demon.radius)
        pygame.draw.circle(self.screen, self.fugitive.color, self.fugitive.position_2d, self.fugitive.radius)

        pygame.display.update()
        return

    def _calc_observations(self):
        rel_position_2d = self.demon.calc_rel_position_2d(self.fugitive.position_2d)
        observations = (rel_position_2d[0], rel_position_2d[1])
        return observations


class TagPlayer:
    def __init__(self, x: int, y: int, radius: int = 5, velocity: int = 0, color: t.Tuple[int, int, int] = (255, 0, 0)) -> None:
        self.x = x
        self.y = y
        self.radius = radius
        self.velocity = velocity
        self.color = color

        return

    def set_position(self, x: int, y: int):
        self.x, self.y = x, y
        return

    def set_randomize_position(self, x_min: int, y_min: int, x_max: int, y_max: int):
        new_x, new_y = random.randint(x_min, x_max), random.randint(y_min, y_max)
        self.x, self.y = new_x, new_y
        return

    def calc_rel_position_2d(self, target: t.Tuple[int, int]):
        x_rel = target[0] - self.x
        y_rel = target[1] - self.y
        return (x_rel, y_rel)

    @property
    def position_2d(self):
        return (self.x, self.y)


if __name__ == "__main__":
    episodes = 10

    env = TagSimpleGame()

    for episode, _ in enumerate(range(episodes), start=1):
        env.reset()
        is_done = False
        reward = 0

        print("==========")
        print(f"episode: {episode}")

        while not is_done:
            env.render(None)
            action = random.randint(0, 4)
            next_observations, reward, is_done, info = env.step(action)
            continue

        print(f"reward: {reward}")
        print()

        continue
