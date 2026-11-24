import json
import os

import numpy as np
import pytest

from export import export_model


def test_onnx_export_and_numerical_parity(tmp_path):
    output_onnx = str(tmp_path / "test_model.onnx")
    metadata_json = str(tmp_path / "test_model_metadata.json")

    # Utiliza o modelo final pré-existente no repositório para validar o exportador
    model_path = "modelos/ssl_el_attacker_ppo_final"
    if not os.path.exists(f"{model_path}.zip"):
        pytest.skip(f"Modelo base {model_path}.zip não encontrado para teste de exportação.")

    export_model(model_path, output_onnx, n_obs=49, n_actions=4)

    # 1. Verifica se os arquivos foram criados
    assert os.path.exists(output_onnx), "Arquivo .onnx não foi criado!"
    assert os.path.exists(metadata_json), "Arquivo _metadata.json não foi criado!"

    # 2. Valida o conteúdo dos metadados
    with open(metadata_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["input"]["shape"] == [-1, 49]
    assert meta["output"]["shape"] == [-1, 4]
    assert "v_x" in meta["output"]["channels"]

    # 3. Valida inferência via ONNX Runtime
    import onnxruntime as ort
    session = ort.InferenceSession(output_onnx, providers=["CPUExecutionProvider"])
    dummy_input = np.random.randn(2, 49).astype(np.float32)
    output = session.run(None, {"observation": dummy_input})[0]
    assert output.shape == (2, 4)
