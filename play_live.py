# Copyright (c) 2022 RobôCIn, 2026 OxeBots
# Distributed under the MIT License (see LICENSE)
"""
Monitor de Treinamento em Tempo Real (Live Player) - OxeBots RL-SSL
Generalizado para qualquer configuração YAML:
Abre a janela gráfica e recarrega novos checkpoints automaticamente assim que forem salvos pelo train.py.
"""

import argparse
import glob
import os
import time

import gymnasium as gym
import yaml
from stable_baselines3 import PPO

from rewards import get_reward_function


def get_latest_checkpoint(checkpoints_dir: str, final_model_path: str | None = None) -> str:
    """Busca o checkpoint mais recente gerado pelo treinamento."""
    candidates = glob.glob(os.path.join(checkpoints_dir, "*.zip"))
    if final_model_path and os.path.exists(f"{final_model_path}.zip"):
        candidates.append(f"{final_model_path}.zip")

    if candidates:
        candidates.sort(key=os.path.getmtime, reverse=True)
        return candidates[0].replace(".zip", "")
    return None


def main():
    parser = argparse.ArgumentParser(description="Monitor de Treino ao Vivo OxeBots RL-SSL")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Caminho do arquivo YAML de configuração (padrão: config/config.yaml)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.01,
        help="Delay em segundos entre os passos (padrão: 0.01)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Erro: Arquivo '{args.config}' não encontrado.")
        return

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    env_cfg = config.get("environment", {})
    mon_cfg = config.get("monitoring", {})
    train_cfg = config.get("training", {})

    is_single = config.get("mode_single", True)
    default_env_id = "SSL-EL-v0" if is_single else "SSL-EL-Coop-v0"
    env_id = env_cfg.get("id") or default_env_id
    checkpoints_dir = mon_cfg.get("checkpoints_dir", "./logs/checkpoints_single/")
    final_model_path = train_cfg.get("final_model_path", None)

    print("=" * 65)
    print("   MONITOR DE TREINAMENTO AO VIVO (PLAY LIVE) - OXEBOTS RL-SSL")
    print(f"   Configuração: {args.config}")
    print(f"   Ambiente: {env_id} | Pasta Monitorada: {checkpoints_dir}")
    print("   Pressione Ctrl + C no terminal para encerrar.")
    print("=" * 65)

    reward_fn = get_reward_function(config)
    field_params = env_cfg.get("field", {})
    set_piece = env_cfg.get("set_piece", "default")

    # Criação do ambiente com renderização gráfica 2D
    try:
        env = gym.make(
            env_id,
            render_mode="human",
            reward_function=reward_fn,
            enable_dribbler=env_cfg.get("enable_dribbler", False),
            enable_kicker=env_cfg.get("enable_kicker", True),
            max_steps=env_cfg.get("max_steps", 600),
            field_params=field_params,
            set_piece=set_piece,
        )
    except TypeError:
        env = gym.make(env_id, render_mode="human")

    # Aguarda o primeiro checkpoint caso nenhum exista
    print(f"\nProcurando checkpoints em: '{checkpoints_dir}'...")
    current_model_path = None
    while True:
        current_model_path = get_latest_checkpoint(checkpoints_dir, final_model_path)
        if current_model_path:
            break
        print("[INFO] Aguardando primeiro checkpoint ser salvo pelo treino (train.py)...")
        time.sleep(5)

    print(f"[OK] Checkpoint inicial detectado: '{current_model_path}'\n")
    model = PPO.load(current_model_path)

    obs, info = env.reset()
    episodes = 0
    goals = 0
    violations = 0

    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(action)

            if args.delay > 0:
                time.sleep(args.delay)

            if terminated or truncated:
                episodes += 1
                if isinstance(info, dict):
                    if info.get("goal", 0) > 0:
                        goals += 1
                        print(f"[GOL] Episódio {episodes}: Gol marcado! (Total gols: {goals})")
                    elif info.get("area_violation", 0) < 0:
                        violations += 1
                        print(f"[FALTA] Episódio {episodes}: Invasão de área! (Total faltas: {violations})")
                    elif info.get("out_of_bounds", 0) < 0:
                        print(f"[FORA] Episódio {episodes}: Fora de campo!")

                # Verifica se um checkpoint mais recente foi salvo
                latest_path = get_latest_checkpoint(checkpoints_dir, final_model_path)
                if latest_path and latest_path != current_model_path:
                    time.sleep(0.3)
                    try:
                        new_model = PPO.load(latest_path)
                        model = new_model
                        current_model_path = latest_path
                        print("\n[ATUALIZACAO AO VIVO] Novo checkpoint carregado!")
                        print(f"   Arquivo: '{latest_path}'\n")
                    except (OSError, RuntimeError, ValueError) as e:
                        print(f"Aviso ao carregar checkpoint: {e}")

                obs, info = env.reset()
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nMonitoramento encerrado pelo usuário.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
