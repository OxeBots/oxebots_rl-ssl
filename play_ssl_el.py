"""
Script de Monitoramento em Tempo Real: Atacante SSL-EL 3v3
abre a janela gráfica  e exibe o progresso do robô durante o treinamento (train_ssl_el_attacker.py).
recarrega novos checkpoints automaticamente assim que forem salvos
"""

import glob
import os
import time
import gymnasium as gym
import rsoccer_gym
from stable_baselines3 import PPO

CHECKPOINTS_DIR = "./modelos/checkpoints_ssl_el_attacker/"
FINAL_MODEL_PATH = "modelos/ssl_el_ppo_final.zip"

def get_latest_checkpoint():
    """Busca o checkpoint ou modelo mais recente gerado pelo treinamento."""
    candidates = glob.glob(os.path.join(CHECKPOINTS_DIR, "*.zip"))
    if os.path.exists(FINAL_MODEL_PATH):
        candidates.append(FINAL_MODEL_PATH)

    if candidates:
        candidates.sort(key=os.path.getmtime, reverse=True)
        return candidates[0].replace(".zip", "")
    
    return None

def main():
    print("=" * 65)
    print("   MONITOR DE TREINAMENTO EM TEMPO REAL - SSL-EL (3v3)")
    print("   Assista ao robô evoluindo enquanto o train_ssl_el_attacker roda!")
    print("   Pressione Ctrl + C no terminal para encerrar.")
    print("=" * 65)

    # 1. Cria o ambiente com renderização gráfica 2D
    env = gym.make("SSL-EL-v0", render_mode="human")

    current_model_path = None
    model = None

    # 2. Aguarda o primeiro checkpoint se nenhum existir
    print("\nProcurando checkpoints em:", CHECKPOINTS_DIR)
    while True:
        current_model_path = get_latest_checkpoint()
        if current_model_path:
            break
        print("⏳ Aguardando primeiro checkpoint ser salvo pelo treinamento (a cada 200k passos)...")
        time.sleep(5)

    print(f"✅ Checkpoint inicial carregado: '{current_model_path}'\n")
    model = PPO.load(current_model_path)

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
                    if info.get("goal", 0) > 0:
                        goals += 1
                        print(f"⚽ Episódio {episodes}: GOL MARCADO! (Total gols: {goals})")
                    elif info.get("goal", 0) < 0:
                        print(f"🥅 Episódio {episodes}: GOL SOFRIDO / CONTRA!")
                    elif info.get("area_violation", 0) < 0:
                        violations += 1
                        print(f"🛑 Episódio {episodes}: Invasão de área! (Total faltas: {violations})")
                    elif info.get("out_of_bounds", 0) < 0:
                        print(f"🚫 Episódio {episodes}: Robô saiu dos limites do campo!")
                    elif info.get("shot_on_goal", 0) > 0:
                        print(f"🎯 Episódio {episodes}: Chute ao gol no alvo defendido!")
                    elif info.get("shot_own_goal", 0) < 0:
                        print(f"⚠️ Episódio {episodes}: Chute contra a própria meta!")
                    else:
                        print(f"⏱️  Episódio {episodes}: Finalizado.")

                # Verifica se um novo checkpoint ou modelo final foi gerado pelo treino
                latest_path = get_latest_checkpoint()
                if latest_path and latest_path != current_model_path:
                    time.sleep(0.5)
                    try:
                        new_model = PPO.load(latest_path)
                        model = new_model
                        current_model_path = latest_path
                        print(f"\n🔄 [ATUALIZAÇÃO EM TEMPO REAL] Novo checkpoint detectado!")
                        print(f"   Carregado: '{latest_path}'\n")
                    except Exception:
                        pass

                obs, info = env.reset()
                time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nMonitoramento encerrado pelo usuário.")
    finally:
        env.close()

if __name__ == "__main__":
    main()
