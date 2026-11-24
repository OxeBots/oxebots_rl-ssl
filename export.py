# Copyright (c) 2022 RobôCIn, 2026 OxeBots
# Distributed under the MIT License (see LICENSE)
"""
Exportador ONNX e Gerador de Metadados para ROS 2 - OxeBots RL-SSL
Converte políticas treinadas (.zip) para ONNX e gera o arquivo de especificação model_metadata.json.
Valida numericamente a paridade entre PyTorch e ONNX Runtime.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from torch import nn


class OnnxablePolicy(nn.Module):
    """Wrapper para extrair apenas a inferência determinística do Ator."""
    def __init__(self, policy):
        super().__init__()
        self.mlp_extractor = policy.mlp_extractor
        self.action_net = policy.action_net

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        latent_pi, _ = self.mlp_extractor(observation)
        return self.action_net(latent_pi)


def export_model(model_path: str, output_path: str, n_obs: int = 49, n_actions: int = 4):
    """Exporta o modelo PPO para ONNX e valida a paridade numérica."""
    import onnx
    import onnxruntime as ort

    print(f"Carregando modelo PyTorch de: '{model_path}'...")
    model = PPO.load(model_path, device="cpu")
    onnx_policy = OnnxablePolicy(model.policy)
    onnx_policy.eval()

    dummy_input = torch.randn(1, n_obs, dtype=torch.float32)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    print(f"Exportando para ONNX: '{output_path}'...")
    torch.onnx.export(
        onnx_policy,
        dummy_input,
        output_path,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch_size"}, "action": {0: "batch_size"}},
    )

    # Garante arquivo auto-contido
    onnx_model = onnx.load(output_path, load_external_data=True)
    onnx.save_model(onnx_model, output_path, save_as_external_data=False)
    data_file = f"{output_path}.data"
    if os.path.exists(data_file):
        os.remove(data_file)

    # Validação de Paridade com ONNX Runtime
    print("Validando paridade numérica PyTorch vs ONNX Runtime...")
    ort_session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    
    test_input = np.random.randn(5, n_obs).astype(np.float32)
    with torch.no_grad():
        torch_out = onnx_policy(torch.from_numpy(test_input)).numpy()
    
    ort_inputs = {ort_session.get_inputs()[0].name: test_input}
    ort_out = ort_session.run(None, ort_inputs)[0]

    max_diff = np.max(np.abs(torch_out - ort_out))
    print(f"Diferença máxima absoluta: {max_diff:.8f}")
    assert max_diff < 1e-4, f"Falha na validação ONNX! Diferença excessiva: {max_diff}"
    print("✅ Paridade matemática 100% validada!")

    # Geração do arquivo model_metadata.json para o ROS 2
    metadata_path = output_path.replace(".onnx", "_metadata.json")
    metadata = {
        "framework": "OxeBots RL-SSL",
        "export_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_file": os.path.basename(output_path),
        "input": {
            "name": "observation",
            "shape": [-1, n_obs],
            "dtype": "float32",
        },
        "output": {
            "name": "action",
            "shape": [-1, n_actions],
            "dtype": "float32",
            "channels": ["v_x", "v_y", "v_theta", "kick_x"],
            "max_v": 1.5,
            "max_w": 5.0,
            "max_kick_x": 3.0,
        },
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"📄 Metadados de deploy salvos em: '{metadata_path}'")


def main():
    parser = argparse.ArgumentParser(description="Exportador ONNX e ROS 2 OxeBots")
    parser.add_argument(
        "--model",
        type=str,
        default="./modelos/ssl_el_single_final",
        help="Caminho do modelo .zip do PPO",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./modelos/ssl_el_single_final.onnx",
        help="Caminho do arquivo .onnx de saída",
    )
    parser.add_argument(
        "--obs-dim",
        type=int,
        default=49,
        help="Dimensão do vetor de observação (padrão: 49)",
    )
    args = parser.parse_args()

    export_model(args.model, args.output, n_obs=args.obs_dim)


if __name__ == "__main__":
    main()
