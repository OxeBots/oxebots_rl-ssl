from collections import deque
import glob
import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor


class SSLMetricsCallback(BaseCallback):
    """
    Callback personalizado para registrar métricas detalhadas de desempenho e
    recompensas decompostas no TensorBoard a cada rollout.
    """
    def __init__(self, stats_window_size: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.stats_window_size = stats_window_size
        self.episode_rewards = deque(maxlen=stats_window_size)
        self.episode_lengths = deque(maxlen=stats_window_size)
        self.goals = deque(maxlen=stats_window_size)
        self.own_goals = deque(maxlen=stats_window_size)
        self.shots = deque(maxlen=stats_window_size)
        self.shot_attempts = deque(maxlen=stats_window_size)
        self.ball_out_offensive = deque(maxlen=stats_window_size)
        self.area_violations = deque(maxlen=stats_window_size)
        self.out_of_bounds = deque(maxlen=stats_window_size)

        # Decomposição de recompensas
        self.move_to_ball = deque(maxlen=stats_window_size)
        self.ball_grad = deque(maxlen=stats_window_size)
        self.push_to_goal = deque(maxlen=stats_window_size)
        self.alignment = deque(maxlen=stats_window_size)
        self.kick_action = deque(maxlen=stats_window_size)
        self.infrared = deque(maxlen=stats_window_size)
        self.energy = deque(maxlen=stats_window_size)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for idx, info in enumerate(infos):
            if dones[idx]:
                # Recompensa total e tamanho do episódio (fornecidos pelo VecMonitor)
                if "episode" in info:
                    self.episode_rewards.append(info["episode"]["r"])
                    self.episode_lengths.append(info["episode"]["l"])

                self.goals.append(1.0 if info.get("goal", 0) > 0 else 0.0)
                self.own_goals.append(1.0 if info.get("goal", 0) < 0 else 0.0)
                self.shots.append(1.0 if info.get("shot_on_goal", 0) > 0 else 0.0)
                self.shot_attempts.append(1.0 if (info.get("shot_attempt", 0) > 0 or info.get("shot_on_goal", 0) > 0) else 0.0)
                self.ball_out_offensive.append(1.0 if info.get("ball_out_offensive", 0) > 0 else 0.0)
                self.area_violations.append(1.0 if info.get("area_violation", 0) < 0 else 0.0)
                self.out_of_bounds.append(1.0 if info.get("out_of_bounds", 0) < 0 else 0.0)

                self.move_to_ball.append(info.get("move_to_ball", 0.0))
                self.ball_grad.append(info.get("ball_grad", 0.0))
                self.push_to_goal.append(info.get("push_to_goal", 0.0))
                self.alignment.append(info.get("alignment", 0.0))
                self.kick_action.append(info.get("kick_action", 0.0))
                self.infrared.append(info.get("infrared", 0.0))
                self.energy.append(info.get("energy", 0.0))
        return True

    def _on_rollout_end(self) -> None:
        if len(self.episode_rewards) > 0:
            self.logger.record("metrics/mean_episode_reward", float(np.mean(self.episode_rewards)))
            self.logger.record("metrics/mean_episode_length", float(np.mean(self.episode_lengths)))

        if len(self.goals) > 0:
            self.logger.record("metrics/goal_rate", float(np.mean(self.goals)))
            self.logger.record("metrics/own_goal_rate", float(np.mean(self.own_goals)))
            self.logger.record("metrics/shot_on_goal_rate", float(np.mean(self.shots)))
            self.logger.record("metrics/shot_attempt_rate", float(np.mean(self.shot_attempts)))
            self.logger.record("metrics/ball_out_offensive_rate", float(np.mean(self.ball_out_offensive)))
            self.logger.record("metrics/area_violation_rate", float(np.mean(self.area_violations)))
            self.logger.record("metrics/out_of_bounds_rate", float(np.mean(self.out_of_bounds)))

            self.logger.record("rewards/move_to_ball_mean", float(np.mean(self.move_to_ball)))
            self.logger.record("rewards/ball_grad_mean", float(np.mean(self.ball_grad)))
            self.logger.record("rewards/push_to_goal_mean", float(np.mean(self.push_to_goal)))
            self.logger.record("rewards/alignment_mean", float(np.mean(self.alignment)))
            self.logger.record("rewards/kick_action_mean", float(np.mean(self.kick_action)))
            self.logger.record("rewards/infrared_mean", float(np.mean(self.infrared)))
            self.logger.record("rewards/energy_penalty_mean", float(np.mean(self.energy)))


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
    try:
        import onnx
        onnx_policy = OnnxablePolicy(model.policy)
        onnx_policy.eval()

        dummy_input = torch.randn(1, 49, dtype=torch.float32)
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

        print(f"modelo ONNX exportado com sucesso em: {save_path}")
    except Exception as e:
        print(f" falha na exportação automática para ONNX: {e}")


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

    # limpeza  dos checkpoints intermediários antigos 
    old_checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.zip"))
    if old_checkpoints:
        print(f"Limpando {len(old_checkpoints)} checkpoints intermediários anteriores...")
        for f in old_checkpoints:
            try:
                os.remove(f)
            except OSError:
                pass

    # criação dos ambientes vetorizados em paralelo (com VecMonitor para logging automático no TensorBoard)
    print(f"\nIniciando {num_envs} processos de simulação em paralelo...")
    vec_env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    env = VecMonitor(vec_env)

    # callbacks periódicos: Checkpoint + Métricas Detalhadas no TensorBoard
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1000, 50_000 // num_envs),
        save_path=checkpoint_dir,
        name_prefix="ppo_ssl_el_attacker"
    )
    metrics_callback = SSLMetricsCallback()
    callbacks = CallbackList([checkpoint_callback, metrics_callback])

    # carrega modelo prévio para continuar melhorando ou inicia um novo
    if os.path.exists(final_model_zip):
        print(f"\n Modelo existente encontrado: '{final_model_zip}'")
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
            ent_coef=0.01,
            verbose=1,
            tensorboard_log="./tensorboard_ssl_el_attacker/"
        )
    else:
        print("\n nenhum modelo final anterior encontrado. Criando novo modelo...")
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
            ent_coef=0.01,
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
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=False
        )
    finally:
        # Garante o fechamento correto dos subprocessos
        env.close()

    # salva o modelo final (.zip e .onnx)
    model.save(final_model_name)
    print(f"\n Treinamento concluído com sucesso!")
    print(f" Modelo RL (.zip) salvo em: {final_model_zip}")
    export_to_onnx(model, final_onnx_path)

    # limpeza APENAS dos checkpoints intermediários temporários
    print("limpando checkpoints intermediários...")
    for f in glob.glob(os.path.join(checkpoint_dir, "*.zip")):
        try:
            os.remove(f)
        except OSError:
            pass
    print("checkpoints intermediários limpos. Modelos principais (.zip e .onnx) preservados!")
    print(f" Para assistir ao robô jogando, execute:")
    print(f"   python play_ssl_el_attacker.py\n")


if __name__ == "__main__":
    main()
