import math
import random
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
import pygame

from rsoccer_gym.Entities import Frame, Robot, Ball, Field
from rsoccer_gym.Render.field import VSSRenderField
from rsoccer_gym.Render.utils import COLORS
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv


class SSLELRenderField(VSSRenderField):
    """renderizada  campo SSL EL (4.5m x 3.0m)."""
    length = 4.5
    width = 3.0
    margin = 0.3
    center_circle_r = 0.5
    penalty_length = 0.50
    penalty_width = 1.350
    goal_width = 0.70
    goal_depth = 0.18
    _scale = 180


class SSLELAttackerEnv(SSLBaseEnv):
    """
    ammbiente SSL-EL  3v3 para Atacante:
    - campo: 4.5m x 3.0m
    - area de pênalti: 1.350m (eixo Y) x 0.50m (eixo X)
    - robôs 3 Azuis vs 3 Amarelos
    - agente: aobô azul 0
    - goleiro amarelo (0): parado no meio da linha do gol
    - demais robôs estáticos 
    - dribbler = False
    - ações: [v_x, v_y, v_theta, kick_x]
    """

    def __init__(self, render_mode=None):
        super().__init__(
            field_type=2,
            n_robots_blue=3,
            n_robots_yellow=3,
            time_step=0.025,
            render_mode=render_mode,
        )

        # ajuste  dimensões  do campo SSL-EL
        self.field.length = 4.5
        self.field.width = 3.0
        self.field.penalty_width = 1.350  # comprimento no eixo Y (1.35m)
        self.field.penalty_length = 0.50  # largura no eixo X (0.50m)
        self.field.goal_width = 0.70
        self.field.goal_depth = 0.18

        # normalização de posições
        self.max_pos = max(
            self.field.width / 2, (self.field.length / 2) + self.field.penalty_length
        )

        # rdenrizador gráfico 
        self.field_renderer = SSLELRenderField()
        self.window_size = self.field_renderer.window_size

        # açoes  [v_x, v_y, v_theta, kick_x] 
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # Espaço de Observação (49 dimensões):
        # - Bola (4): [x, y, v_x, v_y]
        # - 3 Robôs Azuis (8 cada): [x, y, sin(th), cos(th), v_x, v_y, v_th, infrared]
        # - 3 Robôs Amarelos (7 cada): [x, y, sin(th), cos(th), v_x, v_y, v_th]
        n_obs = 4 + (8 * self.n_robots_blue) + (7 * self.n_robots_yellow)
        self.observation_space = gym.spaces.Box(
            low=-self.NORM_BOUNDS,
            high=self.NORM_BOUNDS,
            shape=(n_obs,),
            dtype=np.float32,
        )

        # Limites físicos dos atuadores SSL
        self.max_v = 1.5        # Velocidade linear máxima (m/s)
        self.max_w = 5.0       # Velocidade angular máxima (rad/s)
        self.kick_speed_x = 3.0 # Velocidade máxima do chute frontal (m/s)

        self.previous_ball_potential = None
        self.reward_shaping_total = None
        self.shot_opp_active = False
        self.shot_own_active = False

    def reset(self, *, seed=None, options=None):
        self.previous_ball_potential = None
        self.reward_shaping_total = None
        self.shot_opp_active = False
        self.shot_own_active = False
        return super().reset(seed=seed, options=options)

    def step(self, action):
        observation, reward, terminated, truncated, _ = super().step(action)
        return observation, reward, terminated, truncated, self.reward_shaping_total

    def _get_initial_positions_frame(self) -> Frame:
        """Define onde a bola e cada robô nascem no início de cada episódio."""
        frame = Frame()
        half_len = (self.field.length / 2) - 0.25
        half_wid = (self.field.width / 2) - 0.25

        # bola: posicionada aleatoriamente na intermediária ofensiva/meio
        ball_x = random.uniform(-0.5, 0.6)
        ball_y = random.uniform(-half_wid * 0.7, half_wid * 0.7)
        frame.ball = Ball(x=ball_x, y=ball_y)

        # atacante Azul (id 0): posicionado atrás da bola
        frame.robots_blue[0] = Robot(
            x=random.uniform(-half_len + 0.3, min(-0.15, ball_x - 0.35)),
            y=random.uniform(-half_wid * 0.7, half_wid * 0.7),
            theta=random.uniform(0, 360),
        )

        # dois companheiros azuis (id 1, 2) - Estáticos em posições de apoio
        frame.robots_blue[1] = Robot(x=-1.20, y=0.75, theta=0.0)
        frame.robots_blue[2] = Robot(x=-1.20, y=-0.75, theta=0.0)

        # goleiro Amarelo (id 0): na linha do gol
        gk_x = (self.field.length / 2) - 0.12
        frame.robots_yellow[0] = Robot(
            x=gk_x,
            y=0.0,
            theta=180.0,
        )

        # dois defensores amarelos (id 1, 2) - Estáticos em posições defensivas
        frame.robots_yellow[1] = Robot(x=0.90, y=0.70, theta=180.0)
        frame.robots_yellow[2] = Robot(x=0.90, y=-0.70, theta=180.0)

        return frame

    def _get_commands(self, actions):
        """Converte as ações do agente e gera comandos para os 6 robôs."""
        commands = []

        # comando do Atacante (Azul 0)
        v_x = float(np.clip(actions[0], -1.0, 1.0) * self.max_v)
        v_y = float(np.clip(actions[1], -1.0, 1.0) * self.max_v)
        v_theta = float(np.clip(actions[2], -1.0, 1.0) * self.max_w)
        kick_x = self.kick_speed_x if actions[3] > 0 else 0.0

        commands.append(
            Robot(
                yellow=False,
                id=0,
                v_x=v_x,
                v_y=v_y,
                v_theta=v_theta,
                kick_v_x=kick_x,
                kick_v_z=0.0,
                dribbler=False,  # false para entry
            )
        )

        # companheiros Azuis (1 e 2) parados
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

        # robôs Amarelos (Goleiro 0 e Defensores 1, 2) parados
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

    def _is_inside_penalty_area(self, x: float, y: float, is_yellow_area: bool = True) -> bool:
        """
        Verifica se uma coordenada (x, y) está dentro da área de pênalti de 1.35m x 0.50m.
        """
        half_pen_w = self.field.penalty_width / 2  # 1.350 / 2 = 0.675m
        pen_len = self.field.penalty_length       # 0.50m
        half_field_len = self.field.length / 2    # 4.5 / 2 = 2.25m

        if is_yellow_area:
            # Área adversária (lado positivo do campo)
            in_x = (x > half_field_len - pen_len)
        else:
            # Área própria (lado negativo do campo)
            in_x = (x < -half_field_len + pen_len)

        in_y = abs(y) < half_pen_w
        return in_x and in_y

    def _calculate_reward_and_done(self) -> Tuple[float, bool]:
        """
        FUNÇÃO DE RECOMPENSA (REWARD SHAPING):
        Onde definimos os objetivos, penalidades e regras de término do episódio.
        """
        reward = 0.0
        done = False

        if self.reward_shaping_total is None:
            self.reward_shaping_total = {
                "goal": 0.0,
                "shot_on_goal": 0.0,
                "shot_own_goal": 0.0,
                "area_violation": 0.0,
                "out_of_bounds": 0.0,
                "ball_grad": 0.0,
                "alignment": 0.0,
                "infrared": 0.0,
                "energy": 0.0,
            }

        ball = self.frame.ball
        robot = self.frame.robots_blue[0]
        half_len = self.field.length / 2   # 2.25m
        half_wid = self.field.width / 2    # 1.50m
        goal_w = self.field.goal_width / 2 # 0.35m

        
        # término com Penalidade
        
        # robô atacante saiu dos limites do campo de jogo
        if abs(robot.x) > (half_len + 0.05) or abs(robot.y) > (half_wid + 0.05):
            reward = -5.0
            done = True
            self.reward_shaping_total["out_of_bounds"] -= 5.0
            return reward, done

        # invasão da área adversária (amarela)
        if self._is_inside_penalty_area(robot.x, robot.y, is_yellow_area=True):
            reward = -5.0
            done = True
            self.reward_shaping_total["area_violation"] -= 5.0
            return reward, done

        # invasão da própria área (azul)
        if self._is_inside_penalty_area(robot.x, robot.y, is_yellow_area=False):
            reward = -5.0
            done = True
            self.reward_shaping_total["area_violation"] -= 5.0
            return reward, done

        #eventos terminais

        # gol Válido no Adversário (+15.0)
        if ball.x > half_len and abs(ball.y) < goal_w:
            reward = 15.0
            done = True
            self.reward_shaping_total["goal"] += 15.0
            return reward, done

        # gol Sofrido / Gol Contra (-10.0)
        if ball.x < -half_len and abs(ball.y) < goal_w:
            reward = -10.0
            done = True
            self.reward_shaping_total["goal"] -= 10.0
            return reward, done

        # bola fora dos limites do campo (-1.0)
        if abs(ball.x) > (half_len + 0.1) or abs(ball.y) > (half_wid + 0.1):
            reward = -1.0
            done = True
            return reward, done

        #recompensa por chute ao gol adversario e penalidade ao gol aliado
        dist_robot_ball = math.hypot(robot.x - ball.x, robot.y - ball.y)
        is_near_ball = (dist_robot_ball < 0.25) or robot.infrared

        # chute em direção ao gol adversário (X positivo com velocidade)
        if ball.v_x > 0.8:
            t_opp = (half_len - ball.x) / ball.v_x
            if t_opp > 0:
                y_proj_opp = ball.y + (ball.v_y * t_opp)
                if abs(y_proj_opp) <= (goal_w + 0.06):
                    if is_near_ball and not self.shot_opp_active:
                        self.shot_opp_active = True
                        shot_bonus = 3.0
                        reward += shot_bonus
                        self.reward_shaping_total["shot_on_goal"] += shot_bonus
        elif ball.v_x < 0.3:
            self.shot_opp_active = False

        # chute em direção ao próprio gol / meta aliada (X negativo com velocidade)
        if ball.v_x < -0.8:
            t_own = (-half_len - ball.x) / ball.v_x
            if t_own > 0:
                y_proj_own = ball.y + (ball.v_y * t_own)
                if abs(y_proj_own) <= (goal_w + 0.15):
                    if is_near_ball and not self.shot_own_active:
                        self.shot_own_active = True
                        own_shot_penalty = -3.0
                        reward += own_shot_penalty
                        self.reward_shaping_total["shot_own_goal"] += own_shot_penalty
        elif ball.v_x > -0.3:
            self.shot_own_active = False

        #recompensas contínuas para o ataccente

        goal_target = np.array([half_len, 0.0])
        ball_pos = np.array([ball.x, ball.y])
        robot_pos = np.array([robot.x, robot.y])

        # gradiente de potencial da Bola (levar a bola em direção ao gol)
        dist_ball_to_goal = np.linalg.norm(goal_target - ball_pos)
        ball_potential = -dist_ball_to_goal
        if self.previous_ball_potential is not None:
            grad = (ball_potential - self.previous_ball_potential) / self.time_step
            grad_rw = float(np.clip(grad * 0.1, -1.0, 1.0))
            reward += grad_rw
            self.reward_shaping_total["ball_grad"] += grad_rw
        self.previous_ball_potential = ball_potential

        # posicionamento e alinhamento atrás da bola
        vec_ball_goal = goal_target - ball_pos
        vec_ball_goal /= (np.linalg.norm(vec_ball_goal) + 1e-6)

        # ponto ideal 20cm atrás da bola na reta que liga a bola ao gol
        ideal_robot_pos = ball_pos - (vec_ball_goal * 0.20)
        dist_to_ideal = np.linalg.norm(ideal_robot_pos - robot_pos)
        align_rw = float(np.exp(-2.0 * dist_to_ideal) * 0.1)
        reward += align_rw
        self.reward_shaping_total["alignment"] += align_rw

        # bônus por contato frontal 
        if robot.infrared:
            infra_rw = 0.15
            reward += infra_rw
            self.reward_shaping_total["infrared"] += infra_rw

        # barreira Repulsiva desencorajar chegar muito perto da linha da área adversária
        penalty_line_x = half_len - self.field.penalty_length  # 1.75m
        if robot.x > (penalty_line_x - 0.25) and abs(robot.y) < (self.field.penalty_width / 2 + 0.1):
            dist_near = robot.x - (penalty_line_x - 0.25)
            reward -= 0.15 * dist_near

        # penalidade suave de gasto de energia
        energy_rw = float(1e-4 * (abs(robot.v_x) + abs(robot.v_y) + abs(robot.v_theta)))
        reward -= energy_rw
        self.reward_shaping_total["energy"] -= energy_rw

        return reward, done

    def _frame_to_observations(self) -> np.ndarray:
        obs = [
            self.norm_pos(self.frame.ball.x),
            self.norm_pos(self.frame.ball.y),
            self.norm_v(self.frame.ball.v_x),
            self.norm_v(self.frame.ball.v_y),
        ]

        # 3 robôs Azuis (com sensor infravermelho)
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
