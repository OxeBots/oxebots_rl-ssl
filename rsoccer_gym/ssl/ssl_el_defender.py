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
    """renderizada campo SSL EL (4.5m x 3.0m)."""
    length = 4.5
    width = 3.0
    margin = 0.3
    center_circle_r = 0.5
    penalty_length = 0.50
    penalty_width = 1.350
    goal_width = 0.70
    goal_depth = 0.18
    _scale = 180


class SSLELDefenderEnv(SSLBaseEnv):
    """
    ammbiente SSL-EL 3v3 para Atacante:
    - campo: 4.5m x 3.0m
    - area de pênalti: 1.350m (eixo Y) x 0.50m (eixo X)
    - robôs 3 Azuis vs 3 Amarelos
    - atacante amarelo (0): controlado por uma política PPO congelada
    - defensor azul (0): controlado pelo agente em treinamento
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

        # ajuste dimensões do campo SSL-EL
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

        # rendenrizador gráfico
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
        self.max_v = 2.0        # Velocidade linear máxima do defensor (m/s) (Aumentada)
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

        # atacante Amarelo (id 0): posicionado atrás da bola
        frame.robots_yellow[0] = Robot(
            x=random.uniform(-half_len + 0.3, min(-0.15, ball_x - 0.35)),
            y=random.uniform(-half_wid * 0.7, half_wid * 0.7),
            theta=random.uniform(0, 360),
        )

        # dois companheiros amarelos (id 1, 2) - estáticos em posições de apoio
        frame.robots_yellow[1] = Robot(x=-1.20, y=0.75, theta=0.0)
        frame.robots_yellow[2] = Robot(x=-1.20, y=-0.75, theta=0.0)

        # defensor Azul (id 0): na linha do gol
        gk_x = (self.field.length / 2) - 0.12
        frame.robots_blue[0] = Robot(
            x=gk_x,
            y=0.0,
            theta=180.0,
        )

        # dois companheiros azuis (id 1, 2) - estáticos em posições defensivas
        frame.robots_blue[1] = Robot(x=0.90, y=0.70, theta=180.0)
        frame.robots_blue[2] = Robot(x=0.90, y=-0.70, theta=180.0)

        return frame

    def convert_actions(self, action, angle):
        """Denormalize, clip to absolute max and convert to local"""
        # Denormalize
        v_x = action[0] * self.max_v
        v_y = action[1] * self.max_v
        v_theta = action[2] * self.max_w
        # Convert to local
        v_x, v_y = v_x * np.cos(angle) + v_y * np.sin(angle), -v_x * np.sin(
            angle
        ) + v_y * np.cos(angle)

        # clip by max absolute
        v_norm = np.linalg.norm([v_x, v_y])
        c = v_norm < self.max_v or self.max_v / v_norm
        v_x, v_y = v_x * c, v_y * c

        return v_x, v_y, v_theta

    def _compute_basic_attacker_command(self):
        """
        Comportamento básico do atacante: vai até a bola e chuta em direção
        ao gol do adversário (lado positivo de X).
        Usa controle proporcional simples.
        """
        attacker = self.frame.robots_yellow[0]
        ball = self.frame.ball
        half_len = self.field.length / 2  # 2.25m

        # Alvo do chute: centro do gol adversário (lado +X)
        goal_target = np.array([half_len, 0.0])
        ball_pos = np.array([ball.x, ball.y])
        att_pos = np.array([attacker.x, attacker.y])

        # Vetor bola -> gol e direção
        vec_b2g = goal_target - ball_pos
        dist_b2g = float(np.linalg.norm(vec_b2g))
        dir_b2g = vec_b2g / max(dist_b2g, 1e-6)

        # Ponto alvo: ligeiramente atrás da bola (oposto ao gol)
        approach_offset = 0.10  # 10cm atrás da bola
        target_pos = ball_pos - approach_offset * dir_b2g

        # Controle proporcional de posição (no referencial global)
        Kp_v = 3.0  # ganho de velocidade linear
        vec_to_target = target_pos - att_pos
        dist_to_target = float(np.linalg.norm(vec_to_target))

        vx_global = Kp_v * vec_to_target[0]
        vy_global = Kp_v * vec_to_target[1]

        # Limita a velocidade global do atacante (Reduzida)
        max_v_att = 1.0
        v_norm = math.hypot(vx_global, vy_global)
        if v_norm > max_v_att:
            scale = max_v_att / v_norm
            vx_global *= scale
            vy_global *= scale

        # Converte velocidade global para o referencial local do robô
        theta_rad = np.deg2rad(attacker.theta)
        cos_t = math.cos(theta_rad)
        sin_t = math.sin(theta_rad)
        vx_local = vx_global * cos_t + vy_global * sin_t
        vy_local = -vx_global * sin_t + vy_global * cos_t

        # Controle angular: aponta para o gol através da bola
        desired_angle = math.atan2(dir_b2g[1], dir_b2g[0])
        angle_err = desired_angle - theta_rad
        # Normaliza para [-pi, pi]
        angle_err = (angle_err + math.pi) % (2 * math.pi) - math.pi
        Kp_w = 4.0
        v_theta = float(np.clip(Kp_w * angle_err, -self.max_w, self.max_w))

        # Chuta quando próximo da bola e razoavelmente alinhado
        kick_v_x = 0.0
        if dist_to_target < 0.15 and abs(angle_err) < 0.5:
            kick_v_x = self.kick_speed_x

        return Robot(
            yellow=True,
            id=0,
            v_x=vx_local,
            v_y=vy_local,
            v_theta=v_theta,
            kick_v_x=kick_v_x,
            kick_v_z=0.0,
            dribbler=False,
        )

    def _get_commands(self, action):
        """Gera comandos para o atacante básico e para o defensor (agente RL)."""
        commands = []

        # Comando do defensor Azul (ID 0), controlado pelo PPO em treinamento.
        angle = np.deg2rad(self.frame.robots_blue[0].theta)
        v_x, v_y, v_theta = self.convert_actions(action, angle)

        commands.append(
            Robot(
                yellow=False,
                id=0,
                v_x=v_x,
                v_y=v_y,
                v_theta=v_theta,
                kick_v_x=0.0,
                kick_v_z=0.0,
                dribbler=False,
            )
        )

        # Companheiros Azuis (1 e 2) parados
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

        # Comando do atacante Amarelo (ID 0) — comportamento básico.
        commands.append(self._compute_basic_attacker_command())

        # Companheiros Amarelos (1 e 2) parados
        for i in [1, 2]:
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
        FUNÇÃO DE RECOMPENSA DE DEFESA:
        - Penaliza gols sofridos
        - Premia afastar a bola e defesas bem-sucedidas
        - Premia o posicionamento na linha de defesa (bissetriz)
        - Sobrevivência sem tomar gols
        """
        reward = 0.0
        done = False

        if self.reward_shaping_total is None:
            self.reward_shaping_total = {
                "goal_conceded": 0.0,
                "ball_cleared": 0.0,
                "ball_out": 0.0,
                "area_violation": 0.0,
                "out_of_bounds": 0.0,
                "positioning": 0.0,
                "ball_dist": 0.0,
                "survival": 0.0,
                "energy": 0.0,
            }

        ball = self.frame.ball
        robot = self.frame.robots_blue[0]
        half_len = self.field.length / 2   # 2.25m
        half_wid = self.field.width / 2    # 1.50m
        goal_w = self.field.goal_width / 2 # 0.35m

        # ----------------------------------------------------
        # 1. Eventos Terminais de Defesa
        # ----------------------------------------------------
        # Gol Sofrido (Bola entra no gol Azul: X > 2.25 e |Y| < 0.35)
        if ball.x > half_len and abs(ball.y) < goal_w:
            reward = -50.0
            done = True
            self.reward_shaping_total["goal_conceded"] += reward
            return reward, done

        # Bola Afastada (Bola cruza o meio de campo para o lado Amarelo: X < 0)
        if ball.x < 0.0:
            reward = 40.0
            done = True
            self.reward_shaping_total["ball_cleared"] += reward
            return reward, done

        # Bola Saiu de Campo (Lateral ou Linha de fundo fora do gol)
        # Se saiu de campo sem ser gol, consideramos defesa bem-sucedida
        if abs(ball.y) > half_wid or (ball.x > half_len and abs(ball.y) >= goal_w) or ball.x < -half_len:
            reward = 20.0
            done = True
            self.reward_shaping_total["ball_out"] += reward
            return reward, done

        # ----------------------------------------------------
        # 2. Violação de Regras pelo Defensor
        # ----------------------------------------------------
        # O goleiro pode entrar no próprio gol (até goal_depth), mas não pode sair pelas outras linhas
        is_out_x = robot.x > (half_len + self.field.goal_depth) or robot.x < -(half_len + 0.05)
        is_out_y = abs(robot.y) > (half_wid + 0.05)
        if is_out_x or is_out_y:
            reward = -50.0  # MESMA PENALIDADE DO GOL SOFRIDO! Evita reward hack de "suicídio".
            done = True
            self.reward_shaping_total["out_of_bounds"] -= 50.0
            return reward, done

        # O defensor não pode ir para a área do goleiro amarelo (x < -1.75)
        # Note que is_yellow_area=False significa área no lado negativo (x < -1.75)
        if self._is_inside_penalty_area(robot.x, robot.y, is_yellow_area=False):
            reward = -50.0
            done = True
            self.reward_shaping_total["area_violation"] -= 50.0
            return reward, done

        # ----------------------------------------------------
        # 3. Potenciais Contínuos (Positioning e Ball Distance)
        # ----------------------------------------------------
        goal_center = np.array([half_len, 0.0])
        ball_pos = np.array([ball.x, ball.y])
        robot_pos = np.array([robot.x, robot.y])

        # Posição ideal do goleiro: na linha entre a bola e o gol, um pouco à frente
        vec_g2b = ball_pos - goal_center
        dist_g2b = float(np.linalg.norm(vec_g2b))
        dir_g2b = vec_g2b / max(dist_g2b, 1e-6)
        
        optimal_dist_from_goal = min(0.5, dist_g2b * 0.5)
        optimal_pos = goal_center + optimal_dist_from_goal * dir_g2b
        cur_dist_optimal = float(np.linalg.norm(optimal_pos - robot_pos))

        if self.last_frame is not None:
            last_ball = self.last_frame.ball
            last_robot = self.last_frame.robots_blue[0]
            last_ball_pos = np.array([last_ball.x, last_ball.y])
            last_robot_pos = np.array([last_robot.x, last_robot.y])

            last_vec_g2b = last_ball_pos - goal_center
            last_dist_g2b = float(np.linalg.norm(last_vec_g2b))
            last_dir_g2b = last_vec_g2b / max(last_dist_g2b, 1e-6)
            last_optimal_dist = min(0.5, last_dist_g2b * 0.5)
            last_optimal_pos = goal_center + last_optimal_dist * last_dir_g2b
            last_dist_optimal = float(np.linalg.norm(last_optimal_pos - last_robot_pos))

            # Recompensa por se aproximar da posição ideal
            diff_pos = (last_dist_optimal - cur_dist_optimal) * 3.0
            r_pos = float(np.clip(diff_pos, -1.0, 1.0))
            reward += r_pos
            self.reward_shaping_total["positioning"] += r_pos

            # Recompensa por afastar a bola do gol
            diff_ball = (dist_g2b - last_dist_g2b) * 5.0
            r_ball = float(np.clip(diff_ball, -2.0, 2.0))
            reward += r_ball
            self.reward_shaping_total["ball_dist"] += r_ball

        # ----------------------------------------------------
        # 4. Recompensa de Sobrevivência (tempo) e Custo de Energia
        # ----------------------------------------------------
        survival_rw = 0.01  # Premia o defensor por manter o jogo vivo sem tomar gol
        energy_rw = float(1e-4 * (abs(robot.v_x) + abs(robot.v_y) + abs(robot.v_theta)))
        reward += survival_rw - energy_rw
        self.reward_shaping_total["survival"] += survival_rw
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

