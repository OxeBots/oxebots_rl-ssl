"""Monitoramento em tempo real do treinamento do defensor SSL-EL."""

import glob
import os
import time
import gymnasium as gym
import rsoccer_gym
from stable_baselines3 import PPO

DEFENDER_CHECKPOINTS_DIR = "./modelos/checkpoints_ssl_el_defender/"
DEFENDER_FINAL_MODEL_PATH = "modelos/ssl_el_defender_ppo_final.zip"

def get_latest_model(checkpoints_dir, final_model_path):
    """Retorna o modelo mais recente sem a extensão .zip."""
    candidates = glob.glob(os.path.join(checkpoints_dir, "*.zip"))
    if os.path.exists(final_model_path):
        candidates.append(final_model_path)

    if candidates:
        candidates.sort(key=os.path.getmtime, reverse=True)
        return candidates[0].replace(".zip", "")

    return None

def main():
    print("=" * 65)
    print("   MONITOR DE TREINAMENTO EM TEMPO REAL - DEFENSOR SSL-EL (3v3)")
    print("   O atacante usa comportamento básico (vai para a bola e chuta).")
    print("   Pressione Ctrl + C no terminal para encerrar.")
    print("=" * 65)

    env = gym.make(
        "SSL-EL-Defender-v0",
        render_mode="human",
    )

    current_model_path = None
    model = None

    print("\nProcurando checkpoints do defensor em:", DEFENDER_CHECKPOINTS_DIR)
    while True:
        current_model_path = get_latest_model(
            DEFENDER_CHECKPOINTS_DIR,
            DEFENDER_FINAL_MODEL_PATH,
        )
        if current_model_path:
            break
        print("⏳ Aguardando primeiro checkpoint ser salvo pelo treinamento (a cada 200k passos)...")
        time.sleep(5)

    print(f"✅ Checkpoint inicial carregado: '{current_model_path}'\n")
    model = PPO.load(current_model_path, device='cpu')

    obs, info = env.reset()
    episodes = 0
    goals = 0
    violations = 0

    try:
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                episodes += 1
                if info:
                    if info.get("goal_conceded", 0) < 0:
                        goals += 1
                        print(f"🥅 Episódio {episodes}: GOL SOFRIDO! (Total sofridos: {goals})")
                    elif info.get("ball_cleared", 0) > 0:
                        print(f"✅ Episódio {episodes}: BOLA AFASTADA COM SUCESSO!")
                    elif info.get("ball_out", 0) > 0:
                        print(f"🛡️ Episódio {episodes}: DEFESA! (Bola saiu pela linha de fundo ou lateral)")
                    elif info.get("area_violation", 0) < 0:
                        violations += 1
                        print(f"🛑 Episódio {episodes}: Invasão da área adversária! (Total faltas: {violations})")
                    elif info.get("out_of_bounds", 0) < 0:
                        print(f"🚫 Episódio {episodes}: Robô saiu dos limites do campo!")
                    else:
                        print(f"⏱️  Episódio {episodes}: Finalizado por tempo.")

                # Verifica se um novo checkpoint ou modelo final foi gerado.
                latest_path = get_latest_model(
                    DEFENDER_CHECKPOINTS_DIR,
                    DEFENDER_FINAL_MODEL_PATH,
                )
                if latest_path and latest_path != current_model_path:
                    time.sleep(0.5)
                    new_model = PPO.load(latest_path, device="cpu")
                    model = new_model
                    current_model_path = latest_path
                    print(f"\n🔄 [ATUALIZAÇÃO EM TEMPO REAL] Novo checkpoint detectado!")
                    print(f"   Carregado: '{latest_path}'\n")

                obs, info = env.reset()
                time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nMonitoramento encerrado pelo usuário.")
    finally:
        env.close()

if __name__ == "__main__":
    main()

