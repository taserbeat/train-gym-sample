import gym
from gym import spaces
import pygame
import numpy as np
import random
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy
from rl.agents.dqn import DQNAgent
import os
import pathlib
import argparse

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
    FIELD_X_MIN, FILED_X_MAX = 0, 100  # フィールドのx座標範囲
    FIELD_Y_MIN, FIELD_Y_MAX = 100, 200  # フィールドのy座標範囲

    FPS = 60  # ゲームのフレームレート
    TIME_LIMIT_SEC = 10  # 1ゲームあたりの制限時間 [単位: 秒]

    RADIUS = 10  # 鬼ごっこプレイヤー(円)の半径
    VELOCITY = 10  # 移動速度 [単位: ピクセル/フレーム]

    # 行動の種類
    ACTION_STAY = 0
    ACTION_LEFT = 1
    ACTION_UP = 2
    ACTION_RIGHT = 3
    ACTION_DOWN = 4

    ACTION_NAMES = {
        0: "Stay",
        1: "Left",
        2: "Up",
        3: "Right",
        4: "Down",
    }

    # 行動の個数
    ACTION_SPACE = 5

    def __init__(self) -> None:
        # 学習の設定
        self.action_space = spaces.Discrete(self.ACTION_SPACE)
        self.observation_space = spaces.Box(low=np.float32([0, 0]), high=np.float32([400, 300]))  # type: ignore
        self.reward_range = (-1, 1)

        # ゲーム設定
        pygame.init()
        self.screen = pygame.display.set_mode((400, 400))
        pygame.display.set_caption("鬼ごっこの強化学習")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 30)

        # ゲームの変数
        self.limit_frame = self.FPS * self.TIME_LIMIT_SEC
        self.remain_frame = self.limit_frame
        self.demon = TagPlayer(x=0, y=0, radius=self.RADIUS, velocity=self.VELOCITY, color=(255, 0, 0))
        self.fugitive = TagPlayer(x=0, y=0, radius=self.RADIUS, color=(0, 0, 255))
        self.selected_action = self.ACTION_STAY

        return

    def reset(self):
        self.remain_frame = self.limit_frame
        self.selected_action = self.ACTION_STAY

        self.demon.set_randomize_position(
            x_min=self.FIELD_X_MIN + self.demon.radius,
            x_max=self.FILED_X_MAX - self.demon.radius,
            y_min=self.FIELD_Y_MIN + self.demon.radius,
            y_max=self.FIELD_Y_MAX - self.demon.radius
        )

        self.fugitive.set_randomize_position(
            x_min=self.FIELD_X_MIN + self.fugitive.radius,
            x_max=self.FILED_X_MAX - self.fugitive.radius,
            y_min=self.FIELD_Y_MIN + self.fugitive.radius,
            y_max=self.FIELD_Y_MAX - self.fugitive.radius
        )

        self.screen.fill((0, 0, 0))

        return self._calc_observations()

    def step(self, action):
        self.selected_action = action
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
            # reward = np.float(self.remain_frame / self.limit_frame)
            reward = np.float(1)
            return (self._calc_observations(), reward, True, {})

        # タイムオーバー
        if self.remain_frame <= 0:
            # (観測値, 報酬, True, {})を返す処理
            reward = np.float(-1)
            return (self._calc_observations(), reward, True, {})

        # ゲーム続行中
        return (self._calc_observations(), np.float(-0.01), False, {})

    def render(self, mode):
        self.clock.tick(self.FPS)
        self.screen.fill((0, 0, 0))

        pygame.draw.circle(self.screen, self.demon.color, self.demon.position_2d, self.demon.radius)
        pygame.draw.circle(self.screen, self.fugitive.color, self.fugitive.position_2d, self.fugitive.radius)

        text_time = self.font.render(f"Time: {self.remain_frame}", True, (255, 255, 255))
        self.screen.blit(text_time, (300, 10))

        text_action = self.font.render(f"Action: {self.ACTION_NAMES[self.selected_action]}", True, (255, 255, 255))
        self.screen.blit(text_action, (0, 10))

        pygame.display.update()
        return

    def _calc_observations(self):
        rel_position_2d = self.demon.calc_rel_position_2d(self.fugitive.position_2d)
        observations: np.ndarray = np.float32([rel_position_2d[0], rel_position_2d[1]])  # type: ignore

        return observations

    @property
    def num_observations(self):
        return len(self._calc_observations())


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


# 設定
ROOT_DIR_PATH = str(pathlib.Path(__file__).parent.parent)
MODEL_DIR_PATH = os.path.join(ROOT_DIR_PATH, "models")
MODEl_FILE_PATH = os.path.join(MODEL_DIR_PATH, "tag-simple.h5")


def load_or_create_model(num_observations: int, num_action_space: int, prioritize_load=True) -> Sequential:
    """モデルを読み込み、または新規作成する

    Args:
        num_observations (int): 観測値の種類数
        num_action_space (int): 行動(選択肢)の数
        prioritize_load (bool, optional): Trueの場合はモデルファイルの読み込みを優先する
    """

    model = Sequential([
        Flatten(input_shape=(1, num_observations)),
        Dense(16, activation='relu'),
        Dense(16, activation='relu'),
        Dense(16, activation='relu'),
        Dense(num_action_space, activation='linear')
    ])

    if not prioritize_load:
        return model

    if os.path.exists(MODEl_FILE_PATH):
        model = load_model(MODEl_FILE_PATH)
        print(f"モデルファイルを読み込みました : {MODEl_FILE_PATH}")
        pass

    if not isinstance(model, Sequential):
        raise TypeError(f"モデルの型が'Sequential'ではありません : {type(model)}")

    return model


def create_agent(model: Sequential, num_action_space: int, memory_limit=600, memory_win_length=1):
    """DQNAgentを作成する

    Args:
        model (Sequential): モデル
        num_action_space (int): 行動(選択肢)の数
        memory_limit (int, optional): メモリーの上限
        memory_win_length (int, optional): メモリーのウィンドウサイズ

    Returns:
        DQNAgent (コンパイルは未実行)
    """

    memory = SequentialMemory(limit=memory_limit, window_length=memory_win_length)

    policy = BoltzmannQPolicy()

    dqn = DQNAgent(
        model=model,
        nb_actions=num_action_space,
        gamma=0.98,
        memory=memory,
        nb_steps_warmup=10,
        target_model_update=1e-2,
        policy=policy
    )

    return dqn


def train(episode: t.Optional[int] = None):
    print("=== 学習モード ===")
    print()

    if episode is None:
        episode = 1000

    env = TagSimpleGame()
    env.reset()

    model = load_or_create_model(env.num_observations, env.ACTION_SPACE)

    dqn = create_agent(model, env.ACTION_SPACE)
    dqn.compile(Adam(lr=1e-3), metrics=["mae"])

    dqn.fit(
        env,
        nb_steps=env.FPS * env.TIME_LIMIT_SEC * episode,
        visualize=True,
        verbose=1,
        log_interval=env.FPS * env.TIME_LIMIT_SEC * 10,
    )

    # TODO: モデルの保存は最大報酬を得たときにのみ行うようにできるか検討する
    if not os.path.exists(MODEL_DIR_PATH):
        os.makedirs(MODEL_DIR_PATH, exist_ok=True)
    dqn.model.save(MODEl_FILE_PATH, overwrite=True)
    return


def test(episode: t.Optional[int] = None):
    print("=== テストモード ===")
    print()

    if episode is None:
        episode = 50

    env = TagSimpleGame()
    env.reset()

    model = load_or_create_model(env.num_observations, env.ACTION_SPACE)

    dqn = create_agent(model, env.ACTION_SPACE)
    dqn.compile(Adam(lr=1e-3), metrics=["mae"])

    dqn.test(env, nb_episodes=episode, visualize=True)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", "-M", metavar="MODE", type=str, default="train", help="実行モード 'train' | 'test'")
    parser.add_argument("--episode", "-E", metavar="N", type=int, default=None, help="繰り返すエピソード数")

    args = parser.parse_args()

    mode: str = args.mode
    episode: t.Optional[int] = args.episode

    if mode == "train":
        train(episode=episode)
    elif mode == "test":
        test(episode=episode)
    else:
        print(f"モード: '{mode}' はありません")

    exit(0)
