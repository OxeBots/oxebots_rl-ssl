import glob
import os
import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv


class OnnxablePolicy(nn.Module):
    """Wrapper para extrair apenas a inferência determinística do Ator."""
    def __init__(self, policy):
        super().__init__()
        self.mlp_extractor = policy.mlp_extractor
        self.action_net = policy.action_net

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        latent_pi, _ = self.mlp_extractor(observation)
        return self.action_net(latent_pi)


def export_to_onnx(model, save_path="modelos/ssl_el_attacker.onnx"):
    """Exporta a política treinada para o formato ONNX (otimizado para ROS / C++ / Python)."""
    orig_device = getattr(model, "device", "cpu")
    try:
        import onnx
        onnx_policy = OnnxablePolicy(model.policy).to("cpu")
        onnx_policy.eval()

        dummy_input = torch.randn(1, 49, dtype=torch.float32, device="cpu")
        torch.onnx.export(
            onnx_policy,
            dummy_input,
            save_path,
            input_names=["observation"],
            output_names=["action"],
            dynamic_axes={"observation": {0: "batch_size"}, "action": {0: "batch_size"}}
        )

        # Garante arquivo único auto-contido sem arquivos .data externos
        onnx_model = onnx.load(save_path, load_external_data=True)
        onnx.save_model(onnx_model, save_path, save_as_external_data=False)
        data_file = f"{save_path}.data"
        if os.path.exists(data_file):
            os.remove(data_file)

        print(f"📦 Modelo ONNX exportado com sucesso em: {save_path}")
    except Exception as e:
        print(f"⚠️ Aviso: Falha na exportação automática para ONNX: {e}")
    finally:
        try:
            model.policy.to(orig_device)
        except Exception:
            pass


def make_env(rank: int, seed: int = 42):
    """Cria uma fábrica de ambientes isolada para cada processo paralelo."""
    def _init():
        import rsoccer_gym
        import gymnasium as gym
        env = gym.make("SSL-EL-v0")
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    num_envs = 8  # Número de processos simultâneos (ótimo para CPUs multi-core)
    checkpoint_dir = "./modelos/checkpoints_ssl_el_attacker/"
    model_dir = "./modelos/"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    final_model_name = os.path.join(model_dir, "ssl_el_attacker_ppo_final")
    final_model_zip = f"{final_model_name}.zip"
    final_onnx_path = os.path.join(model_dir, "ssl_el_attacker.onnx")

    print("=" * 65)
    print(f"   TREINAMENTO PARALELO DO ATACANTE SSL-EL (3v3 - Sem Driblador)")
    print(f"   Ambientes Paralelos: {num_envs} processos (SubprocVecEnv)")
    print(f"   Campo: 4.5m x 3.0m | Área: 1.35m x 0.50m")
    print("=" * 65)

    # 1. Limpeza APENAS dos checkpoints intermediários antigos (mantendo o modelo final)
    old_checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.zip"))
    if old_checkpoints:
        print(f"Limpando {len(old_checkpoints)} checkpoints intermediários anteriores...")
        for f in old_checkpoints:
            try:
                os.remove(f)
            except OSError:
                pass

    # 2. Criação dos ambientes vetorizados em paralelo
    print(f"\nIniciando {num_envs} processos de simulação em paralelo...")
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])

    # 3. Salvamento periódico de checkpoints (a cada 50.000 passos totais para monitoramento rápido)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1000, 50_000 // num_envs),
        save_path=checkpoint_dir,
        name_prefix="ppo_ssl_el_attacker"
    )

    # 4. Carrega modelo prévio para continuar melhorando ou inicia um novo
    if os.path.exists(final_model_zip):
        print(f"\n📂 Modelo existente encontrado: '{final_model_zip}'")
        print("   Continuando o treinamento a partir dos pesos existentes para aprimorá-lo...")
        model = PPO.load(
            final_model_name,
            env=env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,
            verbose=1,
            tensorboard_log="./tensorboard_ssl_el_attacker/"
        )
    else:
        print("\n✨ Nenhum modelo final anterior encontrado. Criando novo modelo...")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,
            verbose=1,
            tensorboard_log="./tensorboard_ssl_el_attacker/"
        )

    total_timesteps = 10_000_000
    print(f"\nIniciando treinamento por +{total_timesteps:,} passos...")
    print("Para monitorar o treino em tempo real no navegador:")
    print("  tensorboard --logdir ./tensorboard_ssl_el_attacker/\n")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            progress_bar=True,
            reset_num_timesteps=False
        )
    finally:
        # Garante o fechamento correto dos subprocessos
        env.close()

    # 5. Salva o modelo final (.zip para RL e .onnx para ROS)
    model.save(final_model_name)
    print(f"\n Treinamento concluído com sucesso!")
    print(f" Modelo RL (.zip) salvo em: {final_model_zip}")
    export_to_onnx(model, final_onnx_path)

    # 6. Limpeza APENAS dos checkpoints intermediários temporários
    print("🧹 Limpando checkpoints intermediários...")
    for f in glob.glob(os.path.join(checkpoint_dir, "*.zip")):
        try:
            os.remove(f)
        except OSError:
            pass
    print("✅ Checkpoints intermediários limpos. Modelos principais (.zip e .onnx) preservados!")
    print(f" Para assistir ao robô jogando, execute:")
    print(f"   python play_ssl_el_attacker.py\n")


if __name__ == "__main__":
    main()
