"""
Script Universal de Visualização / Teste de Modelos Treinados
Permite rodar QUALQUER ambiente com QUALQUER modelo salvo via linha de comando.

Exemplos de uso:
  python play.py                                      # Busca e roda o modelo mais recente
  python play.py --env SSL-EL-v0 --model ssl_el_attacker_ppo_final
  python play.py --env VSS-v0 --model vss_atacante_ppo
  python play.py --env SSL-EL-v0                      # Roda sem modelo (ações aleatórias para testar física)
"""

import argparse
import glob
import os
import time
import gymnasium as gym
import rsoccer_gym
from stable_baselines3 import PPO

def find_latest_model(env_name="SSL-EL-v0"):
    """Busca o modelo treinado mais recente disponível dentro da pasta modelos/."""
    # 1. Procura todos os arquivos .zip dentro da pasta modelos/
    model_files = glob.glob("./modelos/**/*.zip", recursive=True) + glob.glob("./modelos/*.zip")
    
    # 2. Fallback caso não haja arquivos em modelos/
    if not model_files:
        model_files = glob.glob("./checkpoints*/*.zip") + glob.glob("./*.zip")
    
    if model_files:
        # Remove duplicados
        model_files = list(dict.fromkeys(model_files))
        # Ordena pelo horário da última modificação (mais recente primeiro)
        model_files.sort(key=os.path.getmtime, reverse=True)
        return model_files[0].replace(".zip", "")
    
    return None

def main():
    parser = argparse.ArgumentParser(description="Visualizador Universal do rSoccer Gym")
    parser.add_argument(
        "--env",
        type=str,
        default="SSL-EL-v0",
        help="ID do ambiente Gymnasium (ex: SSL-EL-v0, VSS-v0, SSLStaticDefenders-v0)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Caminho do arquivo .zip do modelo treinado (sem a extensão .zip)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Executar ações determinísticas da rede (padrão: True)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.01,
        help="Delay em segundos entre os passos para visualização (padrão: 0.01)",
    )
    args = parser.parse_args()

    print("=" * 65)
    print(f"   VISUALIZADOR UNIVERSAL - Ambiente: {args.env}")
    print("   Pressione Ctrl + C no terminal para encerrar.")
    print("=" * 65)

    # 1. Cria o ambiente com renderização gráfica 2D
    try:
        env = gym.make(args.env, render_mode="human")
    except Exception as e:
        print(f"Erro ao carregar o ambiente '{args.env}': {e}")
        return

    # 2. Identifica e carrega o modelo
    model_path = args.model if args.model else find_latest_model(args.env)
    
    # Se o modelo foi passado por nome e reside dentro de modelos/
    if model_path:
        if not os.path.exists(model_path) and not os.path.exists(f"{model_path}.zip"):
            if os.path.exists(os.path.join("modelos", f"{model_path}.zip")):
                model_path = os.path.join("modelos", model_path)
            elif os.path.exists(os.path.join("modelos", model_path)):
                model_path = os.path.join("modelos", model_path)

    if model_path and os.path.exists(f"{model_path}.zip"):
        print(f"\nCarregando modelo treinado: '{model_path}.zip'")
        model = PPO.load(model_path)
        use_trained = True
    elif model_path and os.path.exists(model_path):
        print(f"\nCarregando modelo treinado: '{model_path}'")
        model = PPO.load(model_path)
        use_trained = True
    else:
        print(f"\nNenhum modelo especificado ou encontrado. Executando com ações aleatórias de teste...")
        use_trained = False

    obs, info = env.reset()
    episodes = 0
    goals = 0
    violations = 0

    try:
        while True:
            if use_trained:
                action, _states = model.predict(obs, deterministic=args.deterministic)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)

            if args.delay > 0:
                time.sleep(args.delay)

            if terminated or truncated:
                episodes += 1
                if info:
                    if info.get("goal", 0) > 0 or reward >= 10.0:
                        goals += 1
                        print(f"⚽ Episódio {episodes}: GOL MARCADO! (Total gols: {goals})")
                    elif info.get("area_violation", 0) < 0:
                        violations += 1
                        print(f"🛑 Episódio {episodes}: Invasão de área! (Total faltas: {violations})")
                    else:
                        print(f"⏱️  Episódio {episodes}: Fim de episódio. (Recompensa final: {reward:.2f})")
                
                obs, info = env.reset()
                time.sleep(0.3)
    except KeyboardInterrupt:
        print("\nVisualização encerrada pelo usuário.")
    finally:
        env.close()

if __name__ == "__main__":
    main()
