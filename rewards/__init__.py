from rewards.base_reward import BaseReward
from rewards.reward_single import SingleAgentReward
from rewards.reward_multi import MultiAgentReward


def get_reward_function(config: dict) -> BaseReward:
    """Fábrica que entrega a recompensa certa baseado no booleano mode_single do config.yaml."""
    is_single = config.get("mode_single", True)
    if not is_single or config.get("mode") == "multi-agent":
        return MultiAgentReward(config)
    return SingleAgentReward(config)


__all__ = ["BaseReward", "SingleAgentReward", "MultiAgentReward", "get_reward_function"]
