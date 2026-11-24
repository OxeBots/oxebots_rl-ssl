from typing import Any

# Copyright (c) 2022 RobôCIn, 2026 OxeBots
# Distributed under the MIT License (see LICENSE)
import numpy as np

from rewards.base_reward import BaseReward


class MultiAgentReward(BaseReward):
    """
    Função de Recompensa Cooperativa para 2 Atacantes Azuis em SSL Entry Level.
    Incentiva troca de passes, desmarcação do atacante sem bola e finalização coordenada.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.last_passer_id = None

    def reset(self):
        self.last_passer_id = None

    def calculate(
        self, env, action, last_frame, frame, sent_commands
    ) -> tuple[float, bool, dict[str, float]]:
        reward = 0.0
        done = False

        info = {
            "goal": 0.0,
            "pass_completed": 0.0,
            "support_position": 0.0,
            "area_violation": 0.0,
            "out_of_bounds": 0.0,
            "ball_grad": 0.0,
            "time_cost": 0.0,
        }

        if frame is None or last_frame is None:
            return reward, done, info

        ball = frame.ball
        blue_0 = frame.robots_blue[0]
        blue_1 = frame.robots_blue[1]

        half_len = env.field.length / 2.0
        half_wid = env.field.width / 2.0
        goal_w = env.field.goal_width / 2.0

        # 1. Gol Coletivo
        if ball.x > half_len and abs(ball.y) < goal_w:
            if self.is_enabled("rewards", "goal"):
                goal_val = self.get_reward_val("goal", 60.0)
                reward += goal_val
                info["goal"] += goal_val
            done = True
            return reward, done, info

        # 2. Gol Contra
        if ball.x < -half_len and abs(ball.y) < goal_w:
            if self.is_enabled("penalties", "own_goal"):
                own_goal_val = self.get_penalty_val("own_goal", -25.0)
                reward += own_goal_val
            done = True
            return reward, done, info

        # 3. Violação de área por qualquer um dos atacantes
        penalty_line_x = half_len - env.field.penalty_length
        viol_0 = (blue_0.x > penalty_line_x) and (abs(blue_0.y) < env.field.penalty_width / 2.0)
        viol_1 = (blue_1.x > penalty_line_x) and (abs(blue_1.y) < env.field.penalty_width / 2.0)

        if viol_0 or viol_1:
            if self.is_enabled("penalties", "area_violation"):
                viol_val = self.get_penalty_val("area_violation", -10.0)
                reward += viol_val
                info["area_violation"] += viol_val
            done = True
            return reward, done, info

        # 4. Bola fora de campo
        if abs(ball.x) > (half_len + 0.1) or abs(ball.y) > (half_wid + 0.1):
            if self.is_enabled("penalties", "out_of_bounds"):
                out_val = self.get_penalty_val("out_of_bounds", -2.0)
                reward += out_val
                info["out_of_bounds"] += out_val
            done = True
            return reward, done, info

        # 5. PBRS de avanço da bola até o gol
        if self.is_enabled("rewards", "ball_to_goal"):
            cur_dist_b2g = float(np.linalg.norm(np.array([half_len, 0.0]) - np.array([ball.x, ball.y])))
            last_dist_b2g = float(np.linalg.norm(np.array([half_len, 0.0]) - np.array([last_frame.ball.x, last_frame.ball.y])))
            weight_b2g = self.get_reward_val("ball_to_goal", 4.0)
            diff_b2g = (last_dist_b2g - cur_dist_b2g) * weight_b2g
            r_b2g = float(np.clip(diff_b2g, -2.0, 2.0))
            reward += r_b2g
            info["ball_grad"] += r_b2g

        # 6. Bônus de Passe Concluído
        if self.is_enabled("rewards", "pass_completed") and (
            (last_frame.robots_blue[0].infrared and blue_1.infrared)
            or (last_frame.robots_blue[1].infrared and blue_0.infrared)
        ):
            pass_val = self.get_reward_val("pass_completed", 15.0)
            reward += pass_val
            info["pass_completed"] += pass_val

        # 7. Custo de Tempo
        if self.is_enabled("penalties", "time_cost"):
            t_cost = self.get_penalty_val("time_cost", -0.015)
            reward += t_cost
            info["time_cost"] += t_cost

        return reward, done, info
