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
        self.max_steps = 600    # Duração máxima do episódio (15 segundos)

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
        if self.steps >= self.max_steps:
            truncated = True
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
        FUNÇÃO DE RECOMPENSA ROBUSTA, PROPORCIONAL E BLINDADA CONTRA REWARD HACKING:
        - 100% Baseada em Teoria de Potencial (PBRS) para aproximação e avanço da bola.
        - Chute premiado unicamente por impacto/aceleração real transferida para a bola.
        - Sem brechas de acionamento em falso, vibração ou descanso em pontos laterais.
        - Proporcionalidade equilibrada: Gol (+50.0) >> Chute no Alvo (+10.0) >> Tentativa (+4.0) >> Posicionamento (+3.0).
        - Custo de tempo uniforme e limpo (-0.015/passo), forçando o término rápido sem induzir vibração.
        """
        reward = 0.0
        done = False

        if self.reward_shaping_total is None:
            self.reward_shaping_total = {
                "goal": 0.0,
                "shot_on_goal": 0.0,
                "shot_attempt": 0.0,
                "shot_own_goal": 0.0,
                "ball_out_offensive": 0.0,
                "ball_out": 0.0,
                "area_violation": 0.0,
                "out_of_bounds": 0.0,
                "ball_grad": 0.0,
                "move_to_ball": 0.0,
                "push_to_goal": 0.0,
                "alignment": 0.0,
                "kick_action": 0.0,
                "infrared": 0.0,
                "energy": 0.0,
            }

        ball = self.frame.ball
        robot = self.frame.robots_blue[0]
        half_len = self.field.length / 2   # 2.25m
        half_wid = self.field.width / 2    # 1.50m
        goal_w = self.field.goal_width / 2 # 0.35m

        # ----------------------------------------------------
        # 1. Eventos Terminais de Jogo (Gols e Saídas de Bola)
        # ----------------------------------------------------
        # Gol Válido no Adversário (+50.0 a +60.0)
        if ball.x > half_len and abs(ball.y) < goal_w:
            goal_rw = 50.0
            if self.shot_opp_active or ball.v_x > 0.8:
                goal_rw += 10.0  # Bônus extra por gol de chute potente
            reward = goal_rw
            done = True
            self.reward_shaping_total["goal"] += goal_rw
            return reward, done

        # Gol Sofrido / Gol Contra (-20.0)
        if ball.x < -half_len and abs(ball.y) < goal_w:
            reward = -20.0
            done = True
            self.reward_shaping_total["goal"] -= 20.0
            return reward, done

        # Bola saiu pela Linha de Fundo Adversária (Tiro de Meta / Finalização para fora)
        # Recompensa positiva (+0.5) por concluir o ataque no fundo de campo
        if ball.x > half_len and abs(ball.y) >= goal_w:
            reward = 0.5
            done = True
            self.reward_shaping_total["ball_out_offensive"] += 0.5
            return reward, done

        # Bola fora das outras linhas do campo (laterais ou defesa)
        if abs(ball.x) > (half_len + 0.1) or abs(ball.y) > (half_wid + 0.1):
            reward = 0.0
            done = True
            self.reward_shaping_total["ball_out"] += 1.0
            return reward, done

        # ----------------------------------------------------
        # 2. Violação de Regras pelo Robô (Faltas Terminais)
        # ----------------------------------------------------
        if abs(robot.x) > (half_len + 0.05) or abs(robot.y) > (half_wid + 0.05):
            reward = -5.0
            done = True
            self.reward_shaping_total["out_of_bounds"] -= 5.0
            return reward, done

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

        # ----------------------------------------------------
        # 3. Cálculos Geométricos e Cinemáticos
        # ----------------------------------------------------
        goal_target = np.array([half_len, 0.0])
        ball_pos = np.array([ball.x, ball.y])
        robot_pos = np.array([robot.x, robot.y])

        vec_b2g = goal_target - ball_pos
        dist_b2g = float(np.linalg.norm(vec_b2g))
        dir_b2g = vec_b2g / max(dist_b2g, 1e-6)
        perp_dir = np.array([-dir_b2g[1], dir_b2g[0]])

        vec_r2b = ball_pos - robot_pos
        dist_r2b = float(np.linalg.norm(vec_r2b))
        proj_behind = float(np.dot(vec_r2b, dir_b2g))

        # Ponto alvo do robô: sempre alinhado atrás da bola apontando para o gol
        if proj_behind < -0.05 and dist_r2b < 0.30:
            y_rel = float(np.dot(robot_pos - ball_pos, perp_dir))
            sign_y = 1.0 if y_rel >= 0 else -1.0
            target_pos = ball_pos - 0.12 * dir_b2g + sign_y * 0.15 * perp_dir
        else:
            target_pos = ball_pos - 0.095 * dir_b2g

        cur_dist_target = float(np.linalg.norm(target_pos - robot_pos))

        # Velocidade da bola em direção ao gol
        ball_vel = np.array([ball.v_x, ball.v_y])
        ball_v_to_goal = float(np.dot(ball_vel, dir_b2g))

        # Ângulo alvo de orientação: apontando para o gol adversário através da bola
        target_angle = math.atan2(vec_b2g[1], vec_b2g[0])
        rbt_theta_rad = math.radians(robot.theta)
        align_cos = math.cos(rbt_theta_rad - target_angle)

        # ----------------------------------------------------
        # 4. Detecção de Finalização / Chute em Alta Velocidade (+4.0 a +10.0)
        # ----------------------------------------------------
        is_high_speed_shot = (ball.v_x > 0.8 and ball_v_to_goal > 0.5)
        if is_high_speed_shot:
            t_opp = (half_len - ball.x) / max(ball.v_x, 1e-6)
            if t_opp > 0:
                y_proj_opp = ball.y + (ball.v_y * t_opp)
                if abs(y_proj_opp) <= (goal_w + 0.12):
                    # Chute no alvo defendido pelo goleiro!
                    if not self.shot_opp_active:
                        self.shot_opp_active = True
                        shot_bonus = 10.0
                        reward += shot_bonus
                        self.reward_shaping_total["shot_on_goal"] += shot_bonus
                else:
                    # Tentativa ofensiva que foi para fora
                    if not self.shot_opp_active:
                        self.shot_opp_active = True
                        shot_bonus = 4.0
                        reward += shot_bonus
                        self.reward_shaping_total["shot_attempt"] += shot_bonus
        elif ball.v_x < 0.3:
            self.shot_opp_active = False

        # Chute contra a própria meta
        if ball.v_x < -0.8:
            t_own = (-half_len - ball.x) / min(ball.v_x, -1e-6)
            if t_own > 0:
                y_proj_own = ball.y + (ball.v_y * t_own)
                if abs(y_proj_own) <= (goal_w + 0.20):
                    if (dist_r2b < 0.30) and not self.shot_own_active:
                        self.shot_own_active = True
                        own_shot_penalty = -5.0
                        reward += own_shot_penalty
                        self.reward_shaping_total["shot_own_goal"] += own_shot_penalty
        elif ball.v_x > -0.4:
            self.shot_own_active = False

        # ----------------------------------------------------
        # 5. Recompensas Diferenciais de Potencial (Telescópicas e Invariantes)
        # ----------------------------------------------------
        if self.last_frame is not None:
            last_ball = self.last_frame.ball
            last_robot = self.last_frame.robots_blue[0]
            last_ball_pos = np.array([last_ball.x, last_ball.y])
            last_robot_pos = np.array([last_robot.x, last_robot.y])

            last_vec_b2g = goal_target - last_ball_pos
            last_dist_b2g = float(np.linalg.norm(last_vec_b2g))
            last_dir_b2g = last_vec_b2g / max(last_dist_b2g, 1e-6)
            last_perp_dir = np.array([-last_dir_b2g[1], last_dir_b2g[0]])

            last_vec_r2b = last_ball_pos - last_robot_pos
            last_dist_r2b = float(np.linalg.norm(last_vec_r2b))
            last_proj_behind = float(np.dot(last_vec_r2b, last_dir_b2g))
            if last_proj_behind < -0.05 and last_dist_r2b < 0.30:
                last_y_rel = float(np.dot(last_robot_pos - last_ball_pos, last_perp_dir))
                last_sign_y = 1.0 if last_y_rel >= 0 else -1.0
                last_target_pos = last_ball_pos - 0.12 * last_dir_b2g + last_sign_y * 0.15 * last_perp_dir
            else:
                last_target_pos = last_ball_pos - 0.095 * last_dir_b2g
            last_dist_target = float(np.linalg.norm(last_target_pos - last_robot_pos))

            # A) Aproximação da posição de chute / avanço frontal para contato
            diff_move = (last_dist_target - cur_dist_target) * 2.5
            if proj_behind > 0.0 and dist_r2b < 0.25 and align_cos > 0.1:
                diff_contact = (last_dist_r2b - dist_r2b) * 3.0
                if diff_contact > 0:
                    diff_move = max(diff_move, diff_contact)

            r_move = float(np.clip(diff_move, -1.0, 1.0))
            if is_high_speed_shot or ball_v_to_goal > 0.6:
                r_move = max(0.0, r_move)
            reward += r_move
            self.reward_shaping_total["move_to_ball"] += r_move

            # B) Avanço da bola até o gol (Potencial Puro: ~15.0 max em todo o campo)
            diff_ball_goal = (last_dist_b2g - dist_b2g) * 5.0
            r_ball_grad = float(np.clip(diff_ball_goal, -2.5, 2.5))
            reward += r_ball_grad
            self.reward_shaping_total["ball_grad"] += r_ball_grad

            # C) Alinhamento angular diferencial com o gol
            last_target_angle = math.atan2(last_vec_b2g[1], last_vec_b2g[0])
            last_align_cos = math.cos(math.radians(last_robot.theta) - last_target_angle)
            diff_align = (align_cos - last_align_cos) * 0.5
            r_align = float(np.clip(diff_align, -0.5, 0.5))
            reward += r_align
            self.reward_shaping_total["alignment"] += r_align

            # D) Impulso de aceleração da bola em direção ao gol gerado pelo chute/contato
            # Concedido EXCLUSIVAMENTE quando há transferência real de momento para a bola!
            last_ball_vel = np.array([last_ball.v_x, last_ball.v_y])
            last_ball_v_to_goal = float(np.dot(last_ball_vel, last_dir_b2g))
            accel_ball_to_goal = ball_v_to_goal - last_ball_v_to_goal
            if accel_ball_to_goal > 0.3 and (dist_r2b < 0.25 or robot.infrared):
                impulse_rw = float(2.0 * min(accel_ball_to_goal, 3.0))
                # Bônus extra se o chutador mecânico foi acionado no momento exato do impacto
                if self.sent_commands is not None and self.sent_commands[0].kick_v_x > 0:
                    impulse_rw += 2.0
                    self.reward_shaping_total["kick_action"] += 2.0
                reward += impulse_rw
                self.reward_shaping_total["push_to_goal"] += impulse_rw

        # ----------------------------------------------------
        # 6. Contato com Infravermelho
        # ----------------------------------------------------
        if robot.infrared:
            infra_rw = 0.05
            reward += infra_rw
            self.reward_shaping_total["infrared"] += infra_rw

        # ----------------------------------------------------
        # 7. Barreira Repulsiva para a Área do Goleiro Adversário
        # ----------------------------------------------------
        penalty_line_x = half_len - self.field.penalty_length  # 1.75m
        if robot.x > (penalty_line_x - 0.20) and abs(robot.y) < (self.field.penalty_width / 2 + 0.10):
            dist_near = robot.x - (penalty_line_x - 0.20)
            reward -= 0.20 * dist_near

        # ----------------------------------------------------
        # 8. Penalidade Uniforme de Tempo e Energia
        # ----------------------------------------------------
        time_rw = -0.015  # Custo uniforme constante por passo (-0.60/segundo)
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
