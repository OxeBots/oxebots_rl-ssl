# Copyright (c) 2022 RobôCIn, 2026 OxeBots
# Distributed under the MIT License (see LICENSE)
from rewards.base_reward import BaseReward
from rewards.reward_multi import MultiAgentReward
from rewards.reward_single import SingleAgentReward

REWARD_REGISTRY: dict[str, type[BaseReward]] = {
    "single": SingleAgentReward,
    "attacker": SingleAgentReward,
    "multi": MultiAgentReward,
    "coop": MultiAgentReward,
}


def register_reward(name: str, cls: type[BaseReward]):
    """Registra uma classe de recompensa personalizada para um papel (ex: goleiro)."""
    REWARD_REGISTRY[name.lower()] = cls


def get_reward_function(config: dict) -> BaseReward:
    """Fábrica que entrega a função de recompensa configurada no config.yaml."""
    rewards_cfg = config.get("rewards", {})
    role = str(config.get("role") or rewards_cfg.get("role") or "").lower()

    if role and role in REWARD_REGISTRY:
        return REWARD_REGISTRY[role](config)

    is_single = config.get("mode_single", True)
    if not is_single or config.get("mode") == "multi-agent":
        return MultiAgentReward(config)
    return SingleAgentReward(config)


__all__ = [
    "REWARD_REGISTRY",
    "BaseReward",
    "MultiAgentReward",
    "SingleAgentReward",
    "get_reward_function",
    "register_reward",
]
