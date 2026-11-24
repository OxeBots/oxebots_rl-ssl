# Copyright (c) 2022 RobôCIn, 2026 OxeBots
# Distributed under the MIT License (see LICENSE)
"""
Testes de Regressão contra a branch main.
Garante que todas as interfaces, ambientes herdados e modelos pré-existentes
continuam 100% funcionais e compatíveis sem quebra de contrato.
"""

import os

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3 import PPO

import rsoccer_gym  # noqa: F401


def test_legacy_vss_env_contract():
    """Valida o contrato do ambiente VSS-v0 existente na main."""
    env = gym.make("VSS-v0")
    assert env.observation_space.shape == (40,)
    assert env.action_space.shape == (2,)

    obs, _info = env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (40,)

    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, _step_info = env.step(action)
    assert next_obs.shape == (40,)
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    env.close()


def test_legacy_ssl_static_defenders_contract():
    """Valida o contrato do ambiente SSLStaticDefenders-v0 existente na main."""
    env = gym.make("SSLStaticDefenders-v0")
    assert env.observation_space.shape == (24,)
    assert env.action_space.shape == (5,)

    obs, _info = env.reset(seed=42)
    assert obs.shape == (24,)

    action = env.action_space.sample()
    next_obs, reward, _terminated, _truncated, _ = env.step(action)
    assert next_obs.shape == (24,)
    assert isinstance(reward, (int, float))
    env.close()


def test_legacy_ssl_dribbling_and_challenges():
    """Valida o registro e instanciação dos desafios herdados do rSoccer."""
    for env_id in ["SSLDribbling-v0", "SSLContestedPossession-v0", "SSLPassEndurance-v0"]:
        env = gym.make(env_id)
        assert hasattr(env, "observation_space")
        assert hasattr(env, "action_space")
        env.close()


def test_ssl_el_main_contract_compatibility():
    """
    Garante que SSL-EL-v0 mantém rigorosamente os mesmos formatos da main:
    - Espaço de observação: 49 dimensões contínuas
    - Espaço de ação: 4 dimensões contínuas
    """
    env = gym.make("SSL-EL-v0")
    assert env.observation_space.shape == (49,)
    assert env.action_space.shape == (4,)

    obs, _ = env.reset(seed=42)
    assert obs.shape == (49,)
    assert obs.dtype == np.float32

    # Verifica se os limites físicos de velocidade permanecem coerentes
    raw_env = env.unwrapped
    assert raw_env.max_v == 1.5
    assert raw_env.max_w == 5.0
    assert raw_env.kick_speed_x == 3.0
    env.close()


def test_preexisting_main_model_backward_compatibility():
    """
    Verifica se o modelo pré-treinado existente na main (ssl_el_attacker_ppo_final.zip)
    consegue ser carregado e produzir ações válidas no ambiente atual.
    """
    model_path = "modelos/ssl_el_attacker_ppo_final"
    model_file = f"{model_path}.zip"

    if not os.path.exists(model_file):
        pytest.skip(f"Modelo pré-existente '{model_file}' não encontrado no repositório.")

    model = PPO.load(model_path, device="cpu")
    env = gym.make("SSL-EL-v0")
    obs, _ = env.reset(seed=42)

    # Executa inferência com o modelo da main
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape == (4,)

    # Aplica o passo no ambiente atual
    next_obs, reward, _terminated, _truncated, _info = env.step(action)
    assert next_obs.shape == (49,)
    assert isinstance(reward, (int, float))
    env.close()
