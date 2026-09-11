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
        self.has_touched_ball = False
        self.kick_rewarded_this_contact = False
        self.ball_stopped_steps = 0
        self.last_action = np.zeros(4)
        return super().reset(seed=seed, options=options)

    def step(self, action):
        self.last_action = action
        observation, reward, terminated, truncated, _ = super().step(action)
        return observation, reward, terminated, truncated, self.reward_shaping_total

    def _get_initial_positions_frame(self) -> Frame:

        frame = Frame()
        half_len = (self.field.length / 2) - 0.25
        half_wid = (self.field.width / 2) - 0.25

        # bola: posicionada aleatoriamente na intermediária ofensiva/meio
        ball_x = random.uniform(-0.5, 0.6)
        ball_y = random.uniform(-half_wid * 0.7, half_wid * 0.7)
        frame.ball = Ball(x=ball_x, y=ball_y)

        # limite seguro de spawn: nunca dentro da própria área de pênalti
        # área própria termina em -half_field_len + penalty_length, damos uma margem extra
        min_safe_x = -(self.field.length / 2) + self.field.penalty_length + 0.15

        # atacante Azul (id 0): posicionado atrás da bola, fora da própria área
        spawn_min_x = max(min_safe_x, -half_len + 0.3)
        spawn_max_x = min(-0.15, ball_x - 0.35)
        # salvaguarda: se o range ficar invertido (bola muito perto do fundo), usa um intervalo fixo mínimo
        if spawn_min_x >= spawn_max_x:
            spawn_max_x = spawn_min_x + 0.2

        frame.robots_blue[0] = Robot(
            x=random.uniform(spawn_min_x, spawn_max_x),
            y=random.uniform(-half_wid * 0.7, half_wid * 0.7),
            theta=random.uniform(0, 360),
        )

        # dois companheiros azuis (id 1, 2) - Em posições de ala para recepção de passe
        frame.robots_blue[1] = Robot(x=random.uniform(-0.2, 1.20), y=0.85, theta=180.0)
        frame.robots_blue[2] = Robot(x=random.uniform(-0.2, 1.20), y=-0.85, theta=180.0)

        # goleiro Amarelo (id 0): na linha do gol
        gk_x = (self.field.length / 2) - 0.12
        frame.robots_yellow[0] = Robot(
            x=gk_x,
            y=0.0,
            theta=180.0,
        )

        # dois defensores amarelos (id 1, 2) - Bloqueando a região central da meta
        frame.robots_yellow[1] = Robot(x=1.10, y=0.35, theta=180.0)
        frame.robots_yellow[2] = Robot(x=1.10, y=-0.35, theta=180.0)

        return frame

    def convert_actions(self, action, angle):
        """Desnormaliza e converte velocidades do referencial global para o referencial local do robô."""
        v_x = float(action[0] * self.max_v)
        v_y = float(action[1] * self.max_v)
        v_theta = float(action[2] * self.max_w)

        # Rotação de coordenadas globais para locais
        v_x_local = v_x * np.cos(angle) + v_y * np.sin(angle)
        v_y_local = -v_x * np.sin(angle) + v_y * np.cos(angle)

        # Limitação pela velocidade máxima linear
        v_norm = np.linalg.norm([v_x_local, v_y_local])
        c = 1.0 if v_norm < self.max_v else (self.max_v / (v_norm + 1e-8))
        v_x_local, v_y_local = v_x_local * c, v_y_local * c

        return v_x_local, v_y_local, v_theta

    def _get_commands(self, actions):
        """Converte as ações do agente e gera comandos para os 6 robôs."""
        commands = []

        # comando do Atacante (Azul 0)
        angle = np.deg2rad(self.frame.robots_blue[0].theta)
        v_x, v_y, v_theta = self.convert_actions(actions, angle)
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
        FUNÇÃO DE RECOMPENSA PARA PASSE:
        - Aproximação orientada com a frente do robô voltada para a bola.
        - Eliminação do congelamento: cessação do potencial de aproximação após o primeiro toque.
        - Recompensa contínua de alinhamento angular frontal.
        - Bônus de primeiro toque e chute frontal.
        - Penalidade para chute no ar / vento.
        - Recompensa de velocidade e vetor direcional da bola em direção ao companheiro alvo.
        - Conclusão com raio de recepção realista (0.35m) e término antecipado se o passe parar.
        """
        reward = 0.0
        done = False

        if self.reward_shaping_total is None:
            self.reward_shaping_total = {
                "pass_success": 0.0,
                "area_violation": 0.0,
                "out_of_bounds": 0.0,
                "ball_grad": 0.0,
                "move_to_ball": 0.0,
                "alignment": 0.0,
                "pass_velocity": 0.0,
                "first_touch": 0.0,
                "kick_button": 0.0,
                "kick_air": 0.0,
                "energy": 0.0,
            }

        ball = self.frame.ball
        robot = self.frame.robots_blue[0]
        robot1 = self.frame.robots_blue[1]
        robot2 = self.frame.robots_blue[2]
        half_len = self.field.length / 2   # 2.25m
        half_wid = self.field.width / 2    # 1.50m

        # ----------------------------------------------------
        # 1. Término com Penalidade por Violação de Regras
        # ----------------------------------------------------
        # Robô saiu dos limites do campo
        if abs(robot.x) > (half_len + 0.05) or abs(robot.y) > (half_wid + 0.05):
            reward = -5.0
            done = True
            self.reward_shaping_total["out_of_bounds"] -= 5.0
            return reward, done

        # Invasão de área de pênalti adversária ou própria
        if self._is_inside_penalty_area(robot.x, robot.y, is_yellow_area=True):
            reward = -5.0
            done = True
            self.reward_shaping_total["area_violation"] -= 5.0
            return reward, done

        if self._is_inside_penalty_area(robot.x, robot.y, is_yellow_area=False):
            reward = -5.0
            done = True
            self.reward_shaping_total["area_violation"] -= 5.0
            return reward, done

        # Bola fora de campo
        if abs(ball.x) > (half_len + 0.1) or abs(ball.y) > (half_wid + 0.1):
            reward = -1.0
            done = True
            self.reward_shaping_total["out_of_bounds"] -= 1.0
            return reward, done

        # ----------------------------------------------------
        # 2. Sucesso do Passe (Bola chega na zona de recepção de qualquer aliado)
        # ----------------------------------------------------
        dist_ball_r1 = math.hypot(robot1.x - ball.x, robot1.y - ball.y)
        dist_ball_r2 = math.hypot(robot2.x - ball.x, robot2.y - ball.y)
        pass_success_radius = 0.35  # Raio de domínio/recepção de robôs SSL

        if dist_ball_r1 < pass_success_radius or dist_ball_r2 < pass_success_radius:
            reward = 50.0
            done = True
            self.reward_shaping_total["pass_success"] += 50.0
            return reward, done

        # ----------------------------------------------------
        # 3. Métricas Geométricas e Detecção de Contato Frontal
        # ----------------------------------------------------
        ball_pos = np.array([ball.x, ball.y])
        robot_pos = np.array([robot.x, robot.y])
        robot1_pos = np.array([robot1.x, robot1.y])
        robot2_pos = np.array([robot2.x, robot2.y])

        cur_dist_robot_ball = float(np.linalg.norm(ball_pos - robot_pos))
        cur_dist_ball_teammate = min(dist_ball_r1, dist_ball_r2)
        target_pos = robot1_pos if dist_ball_r1 < dist_ball_r2 else robot2_pos

        # Ângulo e alinhamento do robô com a bola
        vec_to_ball = ball_pos - robot_pos
        ang_to_ball = math.atan2(vec_to_ball[1], vec_to_ball[0])
        rbt_theta_rad = math.radians(robot.theta)
        facing_ball = math.cos(rbt_theta_rad - ang_to_ball)

        # Contato frontal na boca do robô (raio ~0.1115m centro a centro)
        ball_in_mouth = (cur_dist_robot_ball < 0.13) and (facing_ball > 0.6)
        is_touching = robot.infrared or ball_in_mouth

        # Bônus de Primeiro Toque
        if is_touching and not self.has_touched_ball:
            first_touch_rw = 5.0
            reward += first_touch_rw
            self.reward_shaping_total["first_touch"] += first_touch_rw
            self.has_touched_ball = True

        # ----------------------------------------------------
        # 4. Potenciais Contínuos e Dinâmica da Bola
        # ----------------------------------------------------
        if self.last_frame is not None:
            last_ball = self.last_frame.ball
            last_robot = self.last_frame.robots_blue[0]
            last_ball_pos = np.array([last_ball.x, last_ball.y])
            last_robot_pos = np.array([last_robot.x, last_robot.y])

            last_dist_robot_ball = float(np.linalg.norm(last_ball_pos - last_robot_pos))
            last_dist_ball_r1 = float(np.hypot(robot1_pos[0] - last_ball_pos[0], robot1_pos[1] - last_ball_pos[1]))
            last_dist_ball_r2 = float(np.hypot(robot2_pos[0] - last_ball_pos[0], robot2_pos[1] - last_ball_pos[1]))
            last_dist_ball_teammate = min(last_dist_ball_r1, last_dist_ball_r2)

            # A) Aproximação da bola:
            # - Ativo APENAS antes do toque (elimina o medo de afastar a bola ao chutar)
            # - Modulado por facing_ball para impedir aproximação de ré ou de lado
            if not self.has_touched_ball:
                diff_move = (last_dist_robot_ball - cur_dist_robot_ball) * 2.5
                if diff_move > 0:
                    r_move = float(np.clip(diff_move, 0.0, 1.0)) * max(0.0, facing_ball)
                else:
                    r_move = float(np.clip(diff_move, -0.5, 0.0))
                reward += r_move
                self.reward_shaping_total["move_to_ball"] += r_move

            # B) Avanço da bola em direção ao companheiro alvo
            diff_ball_teammate = (last_dist_ball_teammate - cur_dist_ball_teammate) * 4.0
            if abs(diff_ball_teammate) > 1e-4:
                r_ball_grad = float(np.clip(diff_ball_teammate, -2.0, 4.0))
                reward += r_ball_grad
                self.reward_shaping_total["ball_grad"] += r_ball_grad

        # C) Alinhamento contínuo da frente do robô para a bola (antes do contato)
        if not self.has_touched_ball and cur_dist_robot_ball < 0.8:
            align_rw = float(0.02 * max(0.0, facing_ball))
            reward += align_rw
            self.reward_shaping_total["alignment"] += align_rw

        # D) Direção e Velocidade do Passe (assim que a bola ganha velocidade)
        ball_vel = np.array([ball.v_x, ball.v_y])
        ball_speed = float(np.linalg.norm(ball_vel))
        if self.has_touched_ball and ball_speed > 0.2:
            vec_to_target = target_pos - ball_pos
            dist_target = float(np.linalg.norm(vec_to_target)) + 1e-6
            dir_to_target = vec_to_target / dist_target
            vel_proj = float(np.dot(ball_vel, dir_to_target))
            if vel_proj > 0:
                pass_vel_rw = float(np.clip(vel_proj * 0.2, 0.0, 1.0))
                reward += pass_vel_rw
                self.reward_shaping_total["pass_velocity"] += pass_vel_rw

        # E) Chute Frontal e Penalidade para Chutar o Vazio
        if hasattr(self, 'last_action') and self.last_action[3] > 0:
            if is_touching:
                if not self.kick_rewarded_this_contact:
                    kick_rw = 2.0
                    reward += kick_rw
                    self.reward_shaping_total["kick_button"] += kick_rw
                    self.kick_rewarded_this_contact = True
            else:
                kick_air_penalty = -0.02
                reward += kick_air_penalty
                self.reward_shaping_total["kick_air"] += kick_air_penalty
        else:
            if not is_touching:
                self.kick_rewarded_this_contact = False

        # F) Término antecipado se a bola parar após o passe (passe fraco ou desviado)
        if self.has_touched_ball and cur_dist_robot_ball > 0.25:
            if ball_speed < 0.08:
                self.ball_stopped_steps += 1
                if self.ball_stopped_steps > 20:  # ~0.5s sem movimento após o passe
                    done = True
                    reward -= 2.0
                    return reward, done
            else:
                self.ball_stopped_steps = 0

        # G) Penalidade suave de tempo e custo de energia
        time_rw = -0.005
        energy_rw = float(1e-4 * (abs(robot.v_x) + abs(robot.v_y) + abs(robot.v_theta)))
        reward += time_rw - energy_rw
        self.reward_shaping_total["energy"] -= (abs(time_rw) + energy_rw)

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
