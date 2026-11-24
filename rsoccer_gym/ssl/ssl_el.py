# Copyright (c) 2022 RobôCIn, 2026 OxeBots
# Distributed under the MIT License (see LICENSE)
import random

import gymnasium as gym
import numpy as np

from rsoccer_gym.Entities import Ball, Frame, Robot
from rsoccer_gym.Render.field import VSSRenderField
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv


class SSLELRenderField(VSSRenderField):
    """Renderizador gráfico para o campo oficial da SSL Entry Level (4.5m x 3.0m)."""
    length = 4.5
    width = 3.0
    margin = 0.3
    center_circle_r = 0.5
    penalty_length = 0.50
    penalty_width = 1.350
    goal_width = 0.70
    goal_depth = 0.18
    _scale = 180


class SSLELEnv(SSLBaseEnv):
    """
    Ambiente Base SSL Entry Level (SSL-EL) 3v3 - OxeBots RL-SSL:
    - Campo: 4.5m x 3.0m
    - Área de Pênalti: 1.350m (eixo Y) x 0.50m (eixo X)
    - Robôs: 3 Azuis vs 3 Amarelos
    - Agente Controlado: Robô Azul 0
    - Ações Contínuas (4): [v_x, v_y, v_theta, kick_x]
    - Observação (49 dimensões): Bola (4) + 3 Azuis (8 cada) + 3 Amarelos (7 cada)
    - Recompensa: 100% Desacoplada e Externa (definida via rewards/ e YAML).
    """

    def __init__(
        self,
        render_mode=None,
        reward_function=None,
        enable_dribbler=False,
        enable_kicker=True,
        max_steps=600,
    ):
        super().__init__(
            field_type=2,
            n_robots_blue=3,
            n_robots_yellow=3,
            time_step=0.025,
            render_mode=render_mode,
        )

        # Ajuste de dimensões oficiais SSL-EL
        self.field.length = 4.5
        self.field.width = 3.0
        self.field.penalty_width = 1.350
        self.field.penalty_length = 0.50
        self.field.goal_width = 0.70
        self.field.goal_depth = 0.18

        self.max_pos = max(
            self.field.width / 2, (self.field.length / 2) + self.field.penalty_length
        )

        # Renderizador 2D Pygame
        self.field_renderer = SSLELRenderField()
        self.window_size = self.field_renderer.window_size

        # Espaço de Ação: [v_x, v_y, v_theta, kick_x]
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # Espaço de Observação: 49 dimensões
        n_obs = 4 + (8 * self.n_robots_blue) + (7 * self.n_robots_yellow)
        self.observation_space = gym.spaces.Box(
            low=-self.NORM_BOUNDS,
            high=self.NORM_BOUNDS,
            shape=(n_obs,),
            dtype=np.float32,
        )

        # Limites físicos dos atuadores SSL
        self.max_v = 1.5        # Velocidade linear máxima (m/s)
        self.max_w = 5.0        # Velocidade angular máxima (rad/s)
        self.kick_speed_x = 3.0 # Velocidade máxima do chute frontal (m/s)

        # Parâmetros configuráveis
        self.enable_dribbler = enable_dribbler
        self.enable_kicker = enable_kicker
        self.max_steps = max_steps

        # Função de Recompensa Externa Plug-and-Play
        self.reward_function = reward_function
        if self.reward_function is None:
            try:
                from rewards.reward_single import SingleAgentReward
                self.reward_function = SingleAgentReward()
            except ImportError:
                pass

        self.reward_shaping_total = {}

    def reset(self, *, seed=None, options=None):
        self.steps = 0
        self.reward_shaping_total = {}
        if self.reward_function is not None:
            self.reward_function.reset()
        return super().reset(seed=seed, options=options)

    def step(self, action):
        observation, reward, terminated, truncated, _ = super().step(action)
        if self.steps >= self.max_steps:
            truncated = True
        return observation, reward, terminated, truncated, self.reward_shaping_total

    def _frame_to_observations(self) -> np.ndarray:
        obs = [
            self.norm_pos(self.frame.ball.x),
            self.norm_pos(self.frame.ball.y),
            self.norm_v(self.frame.ball.v_x),
            self.norm_v(self.frame.ball.v_y),
        ]

        # 3 robôs Azuis (com sensor infravermelho no ID 0)
        for i in range(self.n_robots_blue):
            r = self.frame.robots_blue[i]
            obs.extend([
                self.norm_pos(r.x),
                self.norm_pos(r.y),
                np.sin(np.deg2rad(r.theta)),
                np.cos(np.deg2rad(r.theta)),
                self.norm_v(r.v_x),
                self.norm_v(r.v_y),
                self.norm_w(r.v_theta),
                1.0 if r.infrared else 0.0,
            ])

        # 3 robôs Amarelos
        for i in range(self.n_robots_yellow):
            r = self.frame.robots_yellow[i]
            obs.extend([
                self.norm_pos(r.x),
                self.norm_pos(r.y),
                np.sin(np.deg2rad(r.theta)),
                np.cos(np.deg2rad(r.theta)),
                self.norm_v(r.v_x),
                self.norm_v(r.v_y),
                self.norm_w(r.v_theta),
            ])

        return np.array(obs, dtype=np.float32)

    def _get_initial_positions_frame(self) -> Frame:
        """Posicionamento inicial limpo e balanceado para o início do episódio."""
        frame = Frame()
        half_len = (self.field.length / 2) - 0.25
        half_wid = (self.field.width / 2) - 0.25

        # Bola: posicionada aleatoriamente na região central/intermediária
        ball_x = random.uniform(-0.5, 0.6)
        ball_y = random.uniform(-half_wid * 0.7, half_wid * 0.7)
        frame.ball = Ball(x=ball_x, y=ball_y)

        # Agente Azul (ID 0): posicionado atrás da bola
        frame.robots_blue[0] = Robot(
            x=random.uniform(-half_len + 0.3, min(-0.15, ball_x - 0.35)),
            y=random.uniform(-half_wid * 0.7, half_wid * 0.7),
            theta=random.uniform(0, 360),
        )

        # Companheiros Azuis (IDs 1 e 2): posições de apoio
        frame.robots_blue[1] = Robot(x=-1.20, y=0.75, theta=0.0)
        frame.robots_blue[2] = Robot(x=-1.20, y=-0.75, theta=0.0)

        # Goleiro Amarelo (ID 0): na linha da meta adversária
        gk_x = (self.field.length / 2) - 0.12
        frame.robots_yellow[0] = Robot(x=gk_x, y=0.0, theta=180.0)

        # Defensores Amarelos (IDs 1 e 2): posições defensivas
        frame.robots_yellow[1] = Robot(x=0.90, y=0.70, theta=180.0)
        frame.robots_yellow[2] = Robot(x=0.90, y=-0.70, theta=180.0)

        return frame

    def convert_actions(self, action, angle):
        """Converte comandos de velocidade globais para o referencial local do robô."""
        v_x = float(action[0] * self.max_v)
        v_y = float(action[1] * self.max_v)
        v_theta = float(action[2] * self.max_w)

        v_x_local = v_x * np.cos(angle) + v_y * np.sin(angle)
        v_y_local = -v_x * np.sin(angle) + v_y * np.cos(angle)

        v_norm = np.linalg.norm([v_x_local, v_y_local])
        c = 1.0 if v_norm < self.max_v else (self.max_v / (v_norm + 1e-8))
        v_x_local, v_y_local = v_x_local * c, v_y_local * c

        return v_x_local, v_y_local, v_theta

    def _get_commands(self, actions):
        """Monta a lista de comandos para os 6 robôs em campo."""
        commands = []

        # Robô Azul 0 (Agente RL)
        angle = np.deg2rad(self.frame.robots_blue[0].theta)
        v_x, v_y, v_theta = self.convert_actions(actions, angle)

        kick_x = 0.0
        if self.enable_kicker and len(actions) > 3 and actions[3] > 0:
            kick_x = self.kick_speed_x
        dribbler = bool(self.enable_dribbler)

        commands.append(
            Robot(
                yellow=False,
                id=0,
                v_x=v_x,
                v_y=v_y,
                v_theta=v_theta,
                kick_v_x=kick_x,
                kick_v_z=0.0,
                dribbler=dribbler,
            )
        )

        # Companheiros Azuis (1 e 2) estáticos
        for i in [1, 2]:
            commands.append(
                Robot(
                    yellow=False,
                    id=i,
                    v_x=0.0,
                    v_y=0.0,
                    v_theta=0.0,
                    kick_v_x=0.0,
                    kick_v_z=0.0,
                    dribbler=False,
                )
            )

        # Robôs Amarelos (0, 1 e 2) estáticos
        for i in range(self.n_robots_yellow):
            commands.append(
                Robot(
                    yellow=True,
                    id=i,
                    v_x=0.0,
                    v_y=0.0,
                    v_theta=0.0,
                    kick_v_x=0.0,
                    kick_v_z=0.0,
                    dribbler=False,
                )
            )

        return commands

    def _calculate_reward_and_done(self) -> tuple[float, bool]:
        """Calcula recompensa e encerramento delegando 100% para o módulo em rewards/."""
        if self.reward_function is not None:
            reward, done, info = self.reward_function.calculate(
                self, self.sent_commands, self.last_frame, self.frame, self.sent_commands
            )
            self.reward_shaping_total = info
            return reward, done

        return 0.0, False
