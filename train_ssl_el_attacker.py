"""
Script de Treinamento: Atacante SSL-EL 3v3 (Sem Driblador)
Algoritmo: PPO (Proximal Policy Optimization) - Stable-Baselines3
"""

import os
import gymnasium as gym
import rsoccer_gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

def main():
    print("=" * 65)
    print("   TREINAMENTO DO ATACANTE SSL-EL (3v3 - Sem Driblador)")
    print("   Campo: 4.5m x 3.0m | Área: 1.35m x 0.50m")
    print("=" * 65)

    # 1. Criação do ambiente registrado
    env = gym.make("SSL-EL-v0")

    # 2. Configuração de salvamento periódico de checkpoints (a cada 200.000 passos)
    os.makedirs("./modelos/checkpoints_ssl_el_attacker/", exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=200_000,
        save_path="./modelos/checkpoints_ssl_el_attacker/",
        name_prefix="ppo_ssl_el_attacker"
    )

    # 3. Hiperparâmetros do PPO
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="./tensorboard_ssl_el_attacker/"
    )

    total_timesteps = 5_000_000
    print(f"\nIniciando treinamento por {total_timesteps:,} passos...")
    print("Para monitorar o treino em tempo real no navegador:")
    print("  tensorboard --logdir ./tensorboard_ssl_el_attacker/\n")

    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )

    # 4. Salva o modelo final treinado
    os.makedirs("./modelos/", exist_ok=True)
    final_model_name = "modelos/ssl_el_attacker_ppo_final"
    model.save(final_model_name)
    print(f"\n Treinamento concluído com sucesso!")
    print(f" Modelo final salvo em: {final_model_name}.zip")
    print(f" Para assistir ao robô jogando, execute:")
    print(f"   python play_ssl_el_attacker.py\n")

if __name__ == "__main__":
    main()
