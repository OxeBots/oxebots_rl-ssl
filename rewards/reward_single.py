import math
import numpy as np
from typing import Dict, Tuple, Any, Optional
from rewards.base_reward import BaseReward


class SingleAgentReward(BaseReward):
    """
    Função de Recompensa Robusta para Atacante/Jogador Único SSL Entry Level.
    Baseada em Teoria de Potencial (PBRS), com blindagem contra reward hacking,
    leitura direta dos booleanos e valores do YAML.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.shot_opp_active = False
        self.shot_own_active = False

    def reset(self):
        self.shot_opp_active = False
        self.shot_own_active = False

    def calculate(
        self, env, action, last_frame, frame, sent_commands
    ) -> Tuple[float, bool, Dict[str, float]]:
        reward = 0.0
        done = False

        info = {
            "goal": 0.0,
            "shot_on_goal": 0.0,
            "shot_attempt": 0.0,
            "shot_own_goal": 0.0,
            "ball_out": 0.0,
            "area_violation": 0.0,
            "out_of_bounds": 0.0,
            "ball_grad": 0.0,
            "move_to_ball": 0.0,
            "push_to_goal": 0.0,
            "alignment": 0.0,
            "infrared": 0.0,
            "energy": 0.0,
        }

        if frame is None or last_frame is None:
            return reward, done, info

        ball = frame.ball
        last_ball = last_frame.ball
        robot = frame.robots_blue[0]
        last_robot = last_frame.robots_blue[0]

        half_len = env.field.length / 2.0
        half_wid = env.field.width / 2.0
        goal_w = env.field.goal_width / 2.0

        # ----------------------------------------------------
        # 1. Eventos Terminais de Jogo (Gols e Saídas)
        # ----------------------------------------------------
        # Gol no Adversário
        if ball.x > half_len and abs(ball.y) < goal_w:
            if self.is_enabled("rewards", "goal"):
                goal_val = self.get_reward_val("goal", 50.0)
                if self.shot_opp_active or ball.v_x > 0.8:
                    goal_val += 10.0  # Bônus por gol de chute potente
                reward = goal_val
                info["goal"] += goal_val
            done = True
            return reward, done, info

        # Gol Contra
        if ball.x < -half_len and abs(ball.y) < goal_w:
            if self.is_enabled("penalties", "own_goal"):
                own_goal_val = self.get_penalty_val("own_goal", -20.0)
                reward = own_goal_val
                info["goal"] += own_goal_val
            done = True
            return reward, done, info

        # Bola fora de campo
        if abs(ball.x) > (half_len + 0.1) or abs(ball.y) > (half_wid + 0.1):
            if self.is_enabled("penalties", "out_of_bounds"):
                out_val = self.get_penalty_val("out_of_bounds", -2.0)
                reward = out_val
                info["ball_out"] += out_val
            done = True
            return reward, done, info

        # ----------------------------------------------------
        # 2. Violação de Regras pelo Robô (Faltas Terminais)
        # ----------------------------------------------------
        # Invasão de Área (adversária ou própria)
        penalty_line_x = half_len - env.field.penalty_length
        is_in_opp_area = (robot.x > penalty_line_x) and (abs(robot.y) < env.field.penalty_width / 2.0)
        is_in_own_area = (robot.x < -penalty_line_x) and (abs(robot.y) < env.field.penalty_width / 2.0)

        if is_in_opp_area or is_in_own_area:
            if self.is_enabled("penalties", "area_violation"):
                viol_val = self.get_penalty_val("area_violation", -10.0)
                reward = viol_val
                info["area_violation"] += viol_val
            done = True
            return reward, done, info

        # Robô fora das linhas laterais
        if abs(robot.x) > (half_len + 0.15) or abs(robot.y) > (half_wid + 0.15):
            if self.is_enabled("penalties", "out_of_bounds"):
                out_val = self.get_penalty_val("out_of_bounds", -2.0)
                reward = out_val
                info["out_of_bounds"] += out_val
            done = True
            return reward, done, info

        # ----------------------------------------------------
        # 3. Chute em Direção ao Gol Adversário
        # ----------------------------------------------------
        cur_ball_pos = np.array([ball.x, ball.y])
        opp_goal_pos = np.array([half_len, 0.0])
        vec_b2g = opp_goal_pos - cur_ball_pos
        dist_b2g = np.linalg.norm(vec_b2g)
        dir_b2g = vec_b2g / max(dist_b2g, 1e-6)

        ball_vel = np.array([ball.v_x, ball.v_y])
        ball_v_to_goal = float(np.dot(ball_vel, dir_b2g))
        ball_speed = float(np.linalg.norm(ball_vel))
        is_high_speed_shot = (ball_speed > 1.2) and (ball_v_to_goal > 0.8 * ball_speed)

        if is_high_speed_shot:
            self.shot_opp_active = True
            if self.is_enabled("rewards", "shot_on_goal"):
                shot_val = self.get_reward_val("shot_on_goal", 10.0)
                shot_rw = float(shot_val * min(ball_speed / 2.5, 1.5))
                reward += shot_rw
                info["shot_on_goal"] += shot_rw

        # ----------------------------------------------------
        # 4. Potencial de Aproximação (PBRS) e Alinhamento
        # ----------------------------------------------------
        robot_pos = np.array([robot.x, robot.y])
        dist_r2b = float(np.linalg.norm(cur_ball_pos - robot_pos))

        last_robot_pos = np.array([last_robot.x, last_robot.y])
        last_ball_pos = np.array([last_ball.x, last_ball.y])
        last_dist_r2b = float(np.linalg.norm(last_ball_pos - last_robot_pos))
        last_vec_b2g = opp_goal_pos - last_ball_pos
        last_dist_b2g = float(np.linalg.norm(last_vec_b2g))

        # A) PBRS: Aproximação do robô até a bola
        if self.is_enabled("rewards", "move_to_ball"):
            weight_m2b = self.get_reward_val("move_to_ball", 2.0)
            diff_move = (last_dist_r2b - dist_r2b) * weight_m2b
            r_move = float(np.clip(diff_move, -1.0, 1.0))
            if is_high_speed_shot or ball_v_to_goal > 0.6:
                r_move = max(0.0, r_move)
            reward += r_move
            info["move_to_ball"] += r_move

        # B) PBRS: Avanço da bola até o gol
        if self.is_enabled("rewards", "ball_to_goal"):
            weight_b2g = self.get_reward_val("ball_to_goal", 5.0)
            diff_b2g = (last_dist_b2g - dist_b2g) * weight_b2g
            r_b2g = float(np.clip(diff_b2g, -2.5, 2.5))
            reward += r_b2g
            info["ball_grad"] += r_b2g

        # C) Alinhamento angular atrás da bola voltado para o gol
        if self.is_enabled("rewards", "alignment"):
            target_angle = math.atan2(vec_b2g[1], vec_b2g[0])
            rbt_th = math.radians(robot.theta)
            align_cos = math.cos(rbt_th - target_angle)

            last_target_angle = math.atan2(last_vec_b2g[1], last_vec_b2g[0])
            last_align_cos = math.cos(math.radians(last_robot.theta) - last_target_angle)
            align_weight = self.get_reward_val("alignment", 0.5)
            diff_align = (align_cos - last_align_cos) * align_weight
            r_align = float(np.clip(diff_align, -0.5, 0.5))
            reward += r_align
            info["alignment"] += r_align

        # D) Contato no sensor infravermelho
        if self.is_enabled("rewards", "infrared") and robot.infrared:
            infra_val = self.get_reward_val("infrared", 0.05)
            reward += infra_val
            info["infrared"] += infra_val

        # ----------------------------------------------------
        # 5. Penalidades Contínuas de Tempo e Energia
        # ----------------------------------------------------
        if self.is_enabled("penalties", "time_cost"):
            t_cost = self.get_penalty_val("time_cost", -0.015)
            reward += t_cost
            info["energy"] += t_cost

        if self.is_enabled("penalties", "energy_cost"):
            e_rate = self.get_penalty_val("energy_cost", 0.0001)
            e_cost = float(e_rate * (abs(robot.v_x) + abs(robot.v_y) + abs(robot.v_theta)))
            reward -= e_cost
            info["energy"] -= e_cost

        return reward, done, info
