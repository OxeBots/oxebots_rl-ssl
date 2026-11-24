from rewards.base_reward import BaseReward
from rewards.reward_single import SingleAgentReward
from rewards.reward_multi import MultiAgentReward


def get_reward_function(config: dict) -> BaseReward:
    """Fábrica para instanciar a função de recompensa correta a partir do YAML."""
    mode = config.get("mode", "single-agent")
    if mode == "multi-agent":
        return MultiAgentReward(config)
    return SingleAgentReward(config)


__all__ = ["BaseReward", "SingleAgentReward", "MultiAgentReward", "get_reward_function"]
