import gymnasium as gym
import numpy as np

import rsoccer_gym  # noqa: F401


def test_multi_env_make():
    env = gym.make("SSL-EL-Coop-v0")
    raw = env.unwrapped
    assert hasattr(raw, "agents")
    assert "blue_0" in raw.agents
    assert "blue_1" in raw.agents
    env.close()


def test_multi_env_reset():
    env = gym.make("SSL-EL-Coop-v0")
    obs_dict, _info_dict = env.reset(seed=42)
    assert isinstance(obs_dict, dict)
    assert "blue_0" in obs_dict
    assert "blue_1" in obs_dict
    assert obs_dict["blue_0"].shape == (35,)
    assert obs_dict["blue_1"].shape == (35,)
    env.close()


def test_multi_env_step():
    env = gym.make("SSL-EL-Coop-v0")
    raw = env.unwrapped
    _obs_dict, _ = env.reset(seed=42)

    action_dict = {
        "blue_0": np.array([0.1, 0.0, 0.0, 0.0], dtype=np.float32),
        "blue_1": np.array([-0.1, 0.0, 0.0, 0.0], dtype=np.float32),
    }

    next_obs, rewards, _dones, _truncs, _infos = raw.step(action_dict)
    assert isinstance(next_obs, dict)
    assert "blue_0" in next_obs and "blue_1" in next_obs
    assert isinstance(rewards, dict)
    assert "blue_0" in rewards and "blue_1" in rewards
    env.close()
