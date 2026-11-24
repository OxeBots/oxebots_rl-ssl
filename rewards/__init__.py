# Copyright (c) 2022 RobôCIn, 2026 OxeBots
# Distributed under the MIT License (see LICENSE)
from rewards.base_reward import BaseReward
from rewards.reward_multi import MultiAgentReward
from rewards.reward_single import SingleAgentReward


def get_reward_function(config: dict) -> BaseReward:
    """Fábrica que entrega a recompensa certa baseado no booleano mode_single do config.yaml."""
    is_single = config.get("mode_single", True)
    if not is_single or config.get("mode") == "multi-agent":
        return MultiAgentReward(config)
    return SingleAgentReward(config)


__all__ = ["BaseReward", "MultiAgentReward", "SingleAgentReward", "get_reward_function"]
