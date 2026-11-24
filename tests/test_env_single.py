import gymnasium as gym
import numpy as np

import rsoccer_gym  # noqa: F401
from rewards import SingleAgentReward


def test_single_env_make_and_spaces():
    env = gym.make("SSL-EL-v0")
    assert env.observation_space.shape == (49,)
    assert env.action_space.shape == (4,)
    env.close()


def test_single_env_reset():
    env = gym.make("SSL-EL-v0")
    obs, info = env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (49,)
    assert obs.dtype == np.float32
    assert isinstance(info, dict)
    env.close()


def test_single_env_step():
    env = gym.make("SSL-EL-v0")
    _obs, _info = env.reset(seed=42)
    action = np.array([0.5, -0.2, 0.0, 1.0], dtype=np.float32)
    next_obs, reward, terminated, truncated, step_info = env.step(action)

    assert next_obs.shape == (49,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(step_info, dict)
    env.close()


def test_single_env_truncation_limit():
    max_steps = 15
    env = gym.make("SSL-EL-v0", max_steps=max_steps)
    env.reset(seed=42)

    truncated = False
    for _ in range(max_steps + 5):
        action = np.zeros(4, dtype=np.float32)
        _, _, term, trunc, _ = env.step(action)
        if trunc or term:
            truncated = trunc
            break

    assert truncated is True, "O ambiente deve truncar quando atingir max_steps"
    env.close()


def test_single_env_custom_reward_injection():
    custom_cfg = {
        "rewards": {"goal": {"enabled": True, "value": 99.0}},
        "penalties": {},
    }
    custom_reward = SingleAgentReward(custom_cfg)
    env = gym.make("SSL-EL-v0", reward_function=custom_reward)
    env.reset(seed=42)
    assert env.unwrapped.reward_function is custom_reward
    env.close()


def test_single_env_default_field_dimensions():
    env = gym.make("SSL-EL-v0")
    raw = env.unwrapped
    assert raw.field.length == 4.5
    assert raw.field.width == 3.0
    assert raw.field.penalty_length == 0.50
    assert raw.field.penalty_width == 1.350
    assert raw.field.goal_width == 0.70
    assert raw.field.goal_depth == 0.18
    env.close()


def test_single_env_division_b_preset():
    env = gym.make("SSL-EL-v0", field_params={"preset": "division_b"})
    raw = env.unwrapped
    assert raw.field.length == 9.0
    assert raw.field.width == 6.0
    assert raw.field.penalty_length == 1.0
    assert raw.field.penalty_width == 2.0
    assert raw.field.goal_width == 1.0
    assert raw.field.goal_depth == 0.18
    obs, _ = env.reset(seed=42)
    assert obs.shape == (49,)
    env.close()


def test_single_env_custom_dimensions():
    env = gym.make(
        "SSL-EL-v0",
        field_params={
            "preset": "custom",
            "length": 5.0,
            "width": 3.5,
            "penalty_length": 0.60,
            "penalty_width": 1.50,
            "goal_width": 0.80,
            "goal_depth": 0.20,
        },
    )
    raw = env.unwrapped
    assert raw.field.length == 5.0
    assert raw.field.width == 3.5
    assert raw.field.penalty_length == 0.60
    assert raw.field.penalty_width == 1.50
    assert raw.field.goal_width == 0.80
    assert raw.field.goal_depth == 0.20
    obs, _ = env.reset(seed=42)
    assert obs.shape == (49,)
    env.close()
