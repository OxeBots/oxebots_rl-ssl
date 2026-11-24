"""
Visualizador Universal de Modelos Treinados - OxeBots RL-SSL
Permite rodar qualquer modelo salvo (.zip ou .onnx) com renderização gráfica 2D e HUD de estatísticas.
"""

import argparse
import glob
import os
import time
import gymnasium as gym
import numpy as np
import yaml
from stable_baselines3 import PPO

import rsoccer_gym
from rewards import get_reward_function


def find_latest_model(search_dir="./modelos"):
    """Busca o modelo treinado mais recente disponível dentro do diretório."""
    model_files = glob.glob(f"{search_dir}/**/*.zip", recursive=True) + glob.glob(f"{search_dir}/*.zip")
    if not model_files:
        model_files = glob.glob("./logs/checkpoints*/*.zip") + glob.glob("./*.zip")

    if model_files:
        model_files = list(dict.fromkeys(model_files))
        model_files.sort(key=os.path.getmtime, reverse=True)
        return model_files[0].replace(".zip", "")
    return None


def main():
    parser = argparse.ArgumentParser(description="Visualizador Universal OxeBots RL-SSL")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Arquivo YAML de configuração (padrão: config/config.yaml)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Caminho do modelo treinado .zip (sem extensão)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="ID do ambiente Gymnasium (sobrescreve o config se fornecido)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.01,
        help="Delay em segundos entre os passos (padrão: 0.01)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="Número de episódios a executar (0 para contínuo)",
    )
    args = parser.parse_args()

    # Carrega config se existir
    config = {}
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    env_cfg = config.get("environment", {})
    env_id = args.env or env_cfg.get("id", "SSL-EL-v0")
    enable_dribbler = env_cfg.get("enable_dribbler", False)
    enable_kicker = env_cfg.get("enable_kicker", True)
    max_steps = env_cfg.get("max_steps", 600)

    print("=" * 65)
    print(f"   VISUALIZADOR UNIVERSAL - Ambiente: {env_id}")
    print(f"   Driblador: {'Ativado' if enable_dribbler else 'Desativado'} | Chutador: {'Ativado' if enable_kicker else 'Desativado'}")
    print("   Pressione Ctrl + C no terminal para encerrar.")
    print("=" * 65)

    reward_fn = get_reward_function(config)

    # Criação do ambiente com renderização gráfica 2D
    try:
        env = gym.make(
            env_id,
            render_mode="human",
            reward_function=reward_fn,
            enable_dribbler=enable_dribbler,
            enable_kicker=enable_kicker,
            max_steps=max_steps,
        )
    except TypeError:
        # Fallback caso o ambiente não suporte kwargs adicionais
        env = gym.make(env_id, render_mode="human")

    # Localização do modelo
    model_path = args.model
    if not model_path:
        train_cfg = config.get("training", {})
        default_target = train_cfg.get("final_model_path", "./modelos/ssl_el_single_final")
        if os.path.exists(f"{default_target}.zip") or os.path.exists(default_target):
            model_path = default_target
        else:
            model_path = find_latest_model()

    model = None
    use_trained = False
    if model_path:
        load_target = model_path if model_path.endswith(".zip") else f"{model_path}.zip"
        if os.path.exists(load_target):
            print(f"\n📂 Carregando modelo treinado: '{load_target}'")
            model = PPO.load(model_path)
            use_trained = True

    if not use_trained:
        print("\n⚠️ Nenhum modelo treinado encontrado. Executando com ações aleatórias de teste...")

    obs, info = env.reset()
    episodes = 0
    goals = 0
    violations = 0

    try:
        while True:
            if use_trained:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)

            if args.delay > 0:
                time.sleep(args.delay)

            if terminated or truncated:
                episodes += 1
                if isinstance(info, dict):
                    if info.get("goal", 0) > 0:
                        goals += 1
                        print(f"⚽ Episódio {episodes}: GOL MARCADO! (Total gols: {goals})")
                    elif info.get("goal", 0) < 0:
                        print(f"🥅 Episódio {episodes}: GOL SOFRIDO / CONTRA!")
                    elif info.get("area_violation", 0) < 0:
                        violations += 1
                        print(f"🛑 Episódio {episodes}: Invasão de área! (Total faltas: {violations})")
                    elif info.get("out_of_bounds", 0) < 0:
                        print(f"🚫 Episódio {episodes}: Robô ou bola fora de campo!")
                    elif info.get("shot_on_goal", 0) > 0:
                        print(f"🎯 Episódio {episodes}: Chute potente a gol no alvo defendido!")
                    else:
                        print(f"⏱️ Episódio {episodes}: Finalizado.")

                if args.episodes > 0 and episodes >= args.episodes:
                    break

                obs, info = env.reset()
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nVisualização encerrada pelo usuário.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
