import pytest
from rewards import get_reward_function, SingleAgentReward, MultiAgentReward
from rewards.base_reward import BaseReward


def test_reward_factory_single_agent():
    cfg = {"mode_single": True, "rewards": {}, "penalties": {}}
    reward_fn = get_reward_function(cfg)
    assert isinstance(reward_fn, SingleAgentReward)
    assert isinstance(reward_fn, BaseReward)


def test_reward_factory_multi_agent():
    cfg = {"mode_single": False, "rewards": {}, "penalties": {}}
    reward_fn = get_reward_function(cfg)
    assert isinstance(reward_fn, MultiAgentReward)
    assert isinstance(reward_fn, BaseReward)


def test_reward_boolean_toggling():
    cfg = {
        "rewards": {
            "goal": {"enabled": True, "value": 50.0},
            "infrared": {"enabled": False, "value": 10.0},
        },
        "penalties": {
            "area_violation": {"enabled": True, "value": -10.0},
            "time_cost": {"enabled": False, "value": -0.015},
        },
    }

    reward_fn = SingleAgentReward(cfg)

    # Ativados devem retornar o valor numérico
    assert reward_fn.get_reward_val("goal") == 50.0
    assert reward_fn.get_penalty_val("area_violation") == -10.0

    # Desativados via enabled: false devem retornar estritamente 0.0
    assert reward_fn.get_reward_val("infrared") == 0.0
    assert reward_fn.get_penalty_val("time_cost") == 0.0

    # Valores não configurados devem retornar default
    assert reward_fn.get_reward_val("inexistente", default=1.0) == 1.0


def test_reward_reset():
    reward_fn = SingleAgentReward()
    reward_fn.shot_opp_active = True
    reward_fn.reset()
    assert reward_fn.shot_opp_active is False
