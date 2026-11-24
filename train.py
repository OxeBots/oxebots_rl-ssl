# Copyright (c) 2022 RobôCIn, 2026 OxeBots
# Distributed under the MIT License (see LICENSE)
"""
Treinador Universal OxeBots RL-SSL
Controlado 100% via arquivo YAML (Single-Agent SB3 PPO ou Multi-Agent MAPPO).
"""

import argparse
import glob
import os

import torch
import yaml

from rewards import get_reward_function


def load_yaml_config(config_path: str) -> dict:
    """Carrega o arquivo YAML de configuração."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Arquivo de configuração '{config_path}' não foi encontrado.")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def export_to_onnx(model, save_path: str, n_obs: int = 49):
    """Exporta a política treinada do PPO para o formato ONNX."""
    try:
        import onnx
        from torch import nn

        class OnnxablePolicy(nn.Module):
            def __init__(self, policy):
                super().__init__()
                self.mlp_extractor = policy.mlp_extractor
                self.action_net = policy.action_net

            def forward(self, observation: torch.Tensor) -> torch.Tensor:
                latent_pi, _ = self.mlp_extractor(observation)
                return self.action_net(latent_pi)

        onnx_policy = OnnxablePolicy(model.policy)
        onnx_policy.eval()

        dummy_input = torch.randn(1, n_obs, dtype=torch.float32)
        torch.onnx.export(
            onnx_policy,
            dummy_input,
            save_path,
            input_names=["observation"],
            output_names=["action"],
            dynamic_axes={"observation": {0: "batch_size"}, "action": {0: "batch_size"}}
        )

        onnx_model = onnx.load(save_path, load_external_data=True)
        onnx.save_model(onnx_model, save_path, save_as_external_data=False)
        data_file = f"{save_path}.data"
        if os.path.exists(data_file):
            os.remove(data_file)

        print(f"[ONNX] Modelo exportado com sucesso em: {save_path}")
    except (OSError, RuntimeError, ValueError) as e:
        print(f"[ONNX] Aviso: Falha na exportação automática para ONNX: {e}")


def make_single_env(env_id: str, config: dict, rank: int, seed: int = 42):
    """Fábrica de ambientes isolada para cada processo com injeção de recompensa."""
    def _init():
        import gymnasium as gym

        env_cfg = config.get("environment", {})
        reward_fn = get_reward_function(config)

        env = gym.make(
            env_id,
            reward_function=reward_fn,
            enable_dribbler=env_cfg.get("enable_dribbler", False),
            enable_kicker=env_cfg.get("enable_kicker", True),
            max_steps=env_cfg.get("max_steps", 600),
        )
        env.reset(seed=seed + rank)
        return env
    return _init


def train_single_agent(config: dict):
    """Pipeline de treinamento para agente único utilizando Stable-Baselines3 PPO."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    env_cfg = config.get("environment", {})
    mon_cfg = config.get("monitoring", {})
    train_cfg = config.get("training", {})
    ppo_cfg = config.get("ppo", {})

    env_id = env_cfg.get("id", "SSL-EL-v0")
    seed = train_cfg.get("seed", 42)
    n_envs = train_cfg.get("n_envs", 8)
    total_timesteps = train_cfg.get("total_timesteps", 5_000_000)
    use_cuda = train_cfg.get("use_cuda", True)
    resume = train_cfg.get("resume_training", True)
    clean_checkpoints = train_cfg.get("clean_old_checkpoints", True)
    auto_export = train_cfg.get("auto_export_onnx", True)
    show_progress = train_cfg.get("progress_bar", True)

    checkpoints_dir = mon_cfg.get("checkpoints_dir", "./logs/checkpoints_single/")
    tensorboard_dir = mon_cfg.get("tensorboard_dir", "./logs/tensorboard_single/")
    save_freq = mon_cfg.get("save_freq_steps", 50000)

    final_model_path = train_cfg.get("final_model_path", "./modelos/ssl_el_single_final")
    final_model_zip = f"{final_model_path}.zip"
    final_onnx_path = f"{final_model_path}.onnx"

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(os.path.dirname(final_model_path) or ".", exist_ok=True)

    print("=" * 70)
    print("   TREINAMENTO SINGLE-AGENT (PPO) - OXEBOTS RL-SSL")
    print(f"   Ambiente: {env_id} | Processos Paralelos: {n_envs}")
    print(f"   TensorBoard: {tensorboard_dir} (acesse http://localhost:6006)")
    print(f"   Checkpoints para play_live: {checkpoints_dir}")
    print("=" * 70)

    # Limpeza de checkpoints antigos se configurado
    if clean_checkpoints:
        old_cps = glob.glob(os.path.join(checkpoints_dir, "*.zip"))
        if old_cps:
            print(f"[INFO] Limpando {len(old_cps)} checkpoints intermediários anteriores...")
            for f in old_cps:
                try:
                    os.remove(f)
                except OSError:
                    pass

    # Criação dos ambientes vetorizados
    print(f"\nIniciando {n_envs} processos de simulação...")
    if n_envs > 1:
        env = SubprocVecEnv([make_single_env(env_id, config, i, seed) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_single_env(env_id, config, 0, seed)])

    # Callback de Checkpoint
    checkpoint_callback = CheckpointCallback(
        save_freq=max(500, save_freq // n_envs),
        save_path=checkpoints_dir,
        name_prefix="ppo_checkpoint"
    )

    device = "cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"
    print(f"Dispositivo de processamento: {device.upper()}")

    # Inicialização ou carregamento de modelo existente
    is_resumed = False
    if resume and os.path.exists(final_model_zip):
        print(f"\n[INFO] Modelo prévio encontrado: '{final_model_zip}'")
        print("   Continuando treinamento a partir dos pesos existentes...")
        model = PPO.load(
            final_model_path,
            env=env,
            learning_rate=ppo_cfg.get("learning_rate", 3e-4),
            n_steps=ppo_cfg.get("n_steps", 1024),
            batch_size=ppo_cfg.get("batch_size", 128),
            n_epochs=ppo_cfg.get("n_epochs", 10),
            gamma=ppo_cfg.get("gamma", 0.99),
            gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
            clip_range=ppo_cfg.get("clip_range", 0.2),
            ent_coef=ppo_cfg.get("ent_coef", 0.005),
            verbose=1,
            tensorboard_log=tensorboard_dir if mon_cfg.get("enable_tensorboard", True) else None,
            device=device,
        )
        is_resumed = True
    else:
        print("\n[INFO] Criando novo modelo PPO...")
        model = PPO(
            "MlpPolicy",
            env=env,
            learning_rate=ppo_cfg.get("learning_rate", 3e-4),
            n_steps=ppo_cfg.get("n_steps", 1024),
            batch_size=ppo_cfg.get("batch_size", 128),
            n_epochs=ppo_cfg.get("n_epochs", 10),
            gamma=ppo_cfg.get("gamma", 0.99),
            gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
            clip_range=ppo_cfg.get("clip_range", 0.2),
            ent_coef=ppo_cfg.get("ent_coef", 0.005),
            verbose=1,
            tensorboard_log=tensorboard_dir if mon_cfg.get("enable_tensorboard", True) else None,
            device=device,
        )

    print(f"\nIniciando treino por +{total_timesteps:,} passos...")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            progress_bar=show_progress,
            reset_num_timesteps=not is_resumed,
        )
    finally:
        env.close()

    # Salva o modelo final
    model.save(final_model_path)
    print("\n[OK] Treinamento concluído com sucesso!")
    print(f"[INFO] Modelo final salvo em: {final_model_zip}")

    # Exportação para ONNX
    if auto_export:
        n_obs = env.observation_space.shape[0] if hasattr(env, "observation_space") else 49
        export_to_onnx(model, final_onnx_path, n_obs=n_obs)

    print("\nPara visualizar o modelo treinado:")
    print(f"  python play.py --model {final_model_path}\n")


def train_multi_agent(config: dict):
    """Pipeline de treinamento Multi-Agent (MAPPO com CTDE)."""
    from mappo.mappo import train_mappo

    print("=" * 70)
    print("   TREINAMENTO MULTI-AGENT (MAPPO / CTDE) - OXEBOTS RL-SSL")
    print("=" * 70)
    train_mappo(config)


def main():
    parser = argparse.ArgumentParser(description="Treinador Universal OxeBots RL-SSL")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Caminho do arquivo YAML de configuração (padrão: config/config.yaml)",
    )
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    is_single = config.get("mode_single", True)

    if not is_single or config.get("mode") == "multi-agent":
        train_multi_agent(config)
    else:
        train_single_agent(config)


if __name__ == "__main__":
    main()
