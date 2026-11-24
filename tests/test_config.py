import os

import yaml


def test_config_file_exists():
    config_path = "config/config.yaml"
    assert os.path.exists(config_path), f"Arquivo '{config_path}' não foi encontrado!"


def test_config_structure():
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert "mode_single" in cfg, "Chave 'mode_single' ausente no config.yaml"
    assert isinstance(cfg["mode_single"], bool), "'mode_single' deve ser booleano (true/false)"

    assert "environment" in cfg, "Seção 'environment' ausente"
    env_cfg = cfg["environment"]
    assert "id" in env_cfg
    assert isinstance(env_cfg.get("enable_dribbler"), bool)
    assert isinstance(env_cfg.get("enable_kicker"), bool)
    assert isinstance(env_cfg.get("max_steps"), int)

    assert "field" in env_cfg, "Seção 'environment.field' ausente"
    field_cfg = env_cfg["field"]
    assert "preset" in field_cfg, "Chave 'preset' ausente no campo"
    for dim in ["length", "width", "penalty_length", "penalty_width", "goal_width", "goal_depth"]:
        assert dim in field_cfg, f"Dimensão '{dim}' ausente no campo"
        assert isinstance(field_cfg[dim], (int, float)), f"Dimensão '{dim}' deve ser numérica"

    assert "rewards" in cfg, "Seção 'rewards' ausente"
    for r_name, r_item in cfg["rewards"].items():
        assert isinstance(r_item, dict), f"Item {r_name} em rewards deve ser dicionário"
        assert "enabled" in r_item, f"Chave 'enabled' ausente em reward {r_name}"
        assert isinstance(r_item["enabled"], bool), f"'enabled' de {r_name} deve ser booleano"
        assert "value" in r_item, f"Chave 'value' ausente em reward {r_name}"
        assert isinstance(r_item["value"], (int, float)), f"'value' de {r_name} deve ser numérico"

    assert "penalties" in cfg, "Seção 'penalties' ausente"
    for p_name, p_item in cfg["penalties"].items():
        assert isinstance(p_item, dict), f"Item {p_name} em penalties deve ser dicionário"
        assert "enabled" in p_item, f"Chave 'enabled' ausente em penalty {p_name}"
        assert isinstance(p_item["enabled"], bool), f"'enabled' de {p_name} deve ser booleano"
        assert "value" in p_item, f"Chave 'value' ausente em penalty {p_name}"
        assert isinstance(p_item["value"], (int, float)), f"'value' de {p_name} deve ser numérico"

    assert "monitoring" in cfg, "Seção 'monitoring' ausente"
    mon_cfg = cfg["monitoring"]
    assert "tensorboard_dir" in mon_cfg
    assert "checkpoints_dir" in mon_cfg

    assert "training" in cfg, "Seção 'training' ausente"
    assert "ppo" in cfg, "Seção 'ppo' ausente"
