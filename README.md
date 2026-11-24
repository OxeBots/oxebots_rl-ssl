# OxeBots RL-SSL

Framework modular de Aprendizado por Reforço (Reinforcement Learning) para futebol de robôs nas categorias RoboCup Small Size League Entry Level (SSL-EL) e Division B.

O projeto foi reestruturado para oferecer desacoplamento total entre o motor físico do simulador e a modelagem matemática das funções de recompensa, permitindo transição rápida entre ligas, customização via arquivo único de configuração e exportação direta para ROS 2 e C++.

---

## Principais Recursos

- **Configuração Unificada (Zero Argumentos CLI)**: Todas as opções de treino, atuadores, hiperparâmetros, recompensas e monitoramento são configuradas em `config/config.yaml`.
- **Alternância Single e Multi-Agent**: Basta alterar `mode_single: true` (PPO via Stable-Baselines3) ou `mode_single: false` (MAPPO com treinamento centralizado e execução descentralizada).
- **Presets de Campo Dinâmicos**: Suporte nativo ao campo oficial da SSL Entry Level (4.5m x 3.0m) e Division B (9.0m x 6.0m), com ajuste automático das posições iniciais e escala de renderização no Pygame.
- **Funções de Recompensa Plug-and-Play**: Recompensas baseadas em Teoria de Potencial (PBRS) isoladas na pasta `rewards/`, com ativação e ponderação orientadas a booleanos.
- **Controle de Atuadores Físicos**: Ativação ou bloqueio do motor de drible (`enable_dribbler`) e solenoide de chute frontal (`enable_kicker`) no arquivo de configuração.
- **Exportação ONNX e Metadados para ROS 2**: Pipeline de exportação automática com validação matemática de paridade numérica via ONNX Runtime e geração de arquivo `_metadata.json`.
- **Monitoramento em Dois Níveis**: Registro de curvas de convergência no TensorBoard e visualização gráfica contínua no Pygame (`play_live.py`) com hot-reload automático de checkpoints.
- **Suíte de Testes e Não-Regressão**: Cobertura completa de testes automatizados com Pytest, garantindo compatibilidade retroativa com modelos e contratos da branch `main`.

---

## Arquitetura do Sistema

```
                      +--------------------+
                      | config/config.yaml |
                      +---------+----------+
                                |
             +------------------+------------------+
             |                                     |
   (mode_single: true)                   (mode_single: false)
             |                                     |
    +--------v--------+                   +--------v--------+
    |   SB3 PPO       |                   |   MAPPO (CTDE)  |
    |  (Single-Agent) |                   |  (Multi-Agent)  |
    +--------+--------+                   +--------+--------+
             |                                     |
             +------------------+------------------+
                                |
                   +------------v------------+
                   |   rsoccer_gym (Física)  |
                   |  (SSL-EL-v0 / Coop-v0)  |
                   +------------+------------+
                                |
                   +------------v------------+
                   |    rewards/ (PBRS/RL)   |
                   +------------+------------+
                                |
             +------------------+------------------+
             |                                     |
    +--------v--------+                   +--------v--------+
    | logs/checkpoints|                   |   export.py     |
    | (play_live.py)  |                   | (ONNX + ROS 2)  |
    +-----------------+                   +-----------------+
```

---

## Instalação e Ambiente

Recomenda-se o uso de Python 3.10 em ambiente Linux:

```bash
# Clone o repositório
git clone https://github.com/OxeBots/oxebots_rl-ssl.git
cd oxebots_rl-ssl

# Crie e ative o ambiente virtual
python3.10 -m venv venv
source venv/bin/activate

# Atualize o pip e instale os pacotes em modo editável
pip install --upgrade pip
pip install -e .

# Instale dependências de desenvolvimento e teste
pip install pytest ruff
```

---

## Configuração (`config/config.yaml`)

O arquivo `config/config.yaml` controla todo o comportamento do framework:

### Alternar Modo de Aprendizado

```yaml
# true: Atacante individual com Stable-Baselines3 PPO
# false: Cooperação entre dois atacantes com MAPPO (CTDE)
mode_single: true
```

### Selecionar Tamanho do Campo (Entry Level vs Division B)

```yaml
environment:
  max_steps: 600
  enable_dribbler: false
  enable_kicker: true

  field:
    # Opções: "entry" (4.5m x 3.0m) | "division_b" (9.0m x 6.0m) | "custom"
    preset: "entry"
    length: 4.5
    width: 3.0
    penalty_length: 0.50
    penalty_width: 1.350
    goal_width: 0.70
    goal_depth: 0.18
```

Para migrar o treinamento para o campo oficial da Division B (regular), basta alterar `preset` para `"division_b"`. O simulador e o renderizador gráfico adaptarão dimensões, posições e escala automaticamente.

### Habilitar e Ajustar Recompensas

Cada termo possui ativação independente via booleano e valor ponderado:

```yaml
rewards:
  goal:
    enabled: true
    value: 50.0
  shot_on_goal:
    enabled: true
    value: 10.0
  move_to_ball:
    enabled: true
    value: 2.0
  ball_to_goal:
    enabled: true
    value: 5.0
  alignment:
    enabled: true
    value: 0.5
  infrared:
    enabled: true
    value: 0.05
  pass_completed:
    enabled: true
    value: 15.0

penalties:
  area_violation:
    enabled: true
    value: -10.0
  out_of_bounds:
    enabled: true
    value: -2.0
  own_goal:
    enabled: true
    value: -20.0
  time_cost:
    enabled: true
    value: -0.015
  energy_cost:
    enabled: true
    value: 0.0001
```

---

## Execução e Treinamento

### 1. Iniciar o Treinamento

O comando abaixo lê o arquivo `config/config.yaml` e inicia o pipeline configurado:

```bash
python train.py
```

Caso queira usar um arquivo de configuração alternativo:

```bash
python train.py --config caminho/para/outro_config.yaml
```

### 2. Monitoramento Visual em Tempo Real (Live Player)

Em um segundo terminal, execute o monitor de checkpoints. A janela do Pygame abrirá e recarregará os pesos em tempo real sempre que um novo checkpoint for salvo pelo `train.py`:

```bash
python play_live.py
```

### 3. Acompanhamento no TensorBoard

Visualize curvas de recompensa média, perdas de valor e entropia da política:

```bash
tensorboard --logdir logs/tensorboard
```

Acesse o painel em: `http://localhost:6006`.

### 4. Visualização de Modelos Concluídos

Para executar e inspecionar um modelo já treinado:

```bash
# Executa o modelo mais recente encontrado em modelos/
python play.py

# Ou especifique o caminho explicitamente
python play.py --model modelos/ssl_el_final
```

---

## Exportação ONNX e Integração com ROS 2 / C++

O script `export.py` converte a rede neural treinada para o padrão aberto ONNX e gera o arquivo de metadados:

```bash
python export.py --model modelos/ssl_el_final --output modelos/ssl_el_final.onnx
```

### Artefatos Gerados

1. **`ssl_el_final.onnx`**: Grafo determinístico otimizado para inferência rápida em tempo real (compatível com `onnxruntime` em Python e C++).
2. **`ssl_el_final_metadata.json`**: Especificação técnica contendo a ordem das dimensões de observação e mapeamento de canais de controle (`v_x`, `v_y`, `v_theta`, `kick_x`).

Durante a exportação, o pipeline executa automaticamente testes com vetores aleatórios e exige paridade matemática com tolerância inferior a `1e-4` entre a saída do PyTorch e do ONNX Runtime.

---

## Testes Automatizados e Qualidade de Código

O repositório adota testes unitários rigorosos e verificação de não-regressão contra a branch `main`:

```bash
# Executar suíte de testes com relatório detalhado
pytest tests/ -v

# Verificar conformidade de linter e formatação (Ruff)
ruff check .
```

### Cobertura de Testes (`tests/`)

- **`test_config.py`**: Validação estrutural do `config.yaml`, integridade de tipos numéricos e booleanos.
- **`test_env_single.py`**: Espaços de observação (49 dimensões), espaços de ação contínua (4 dimensões), limites de truncamento, injeção de recompensa desacoplada e troca de presets de campo (`entry`, `division_b`, `custom`).
- **`test_env_multi.py`**: Ambiente multi-agente cooperativo (`SSL-EL-Coop-v0`), observações locais (35 dimensões) e alternância de campo.
- **`test_rewards.py`**: Funcionamento da fábrica `get_reward_function`, isolamento dos termos de recompensa e alternância de flags booleanas.
- **`test_export.py`**: Validação ponta a ponta da exportação ONNX, conferência do JSON de metadados e paridade numérica.
- **`test_regression.py`**: Garantia de não-regressão contratual contra a branch `main` (ambientes legados `VSS-v0`, `SSLStaticDefenders-v0`, desafios SSL e carregamento do modelo pré-existente `modelos/ssl_el_attacker_ppo_final.zip`).

---

## Estrutura do Repositório

```
oxebots_rl-ssl/
├── config/
│   └── config.yaml               # Arquivo central unificado de configuração
├── rewards/
│   ├── __init__.py               # Fábrica seletora de recompensas (single vs multi)
│   ├── base_reward.py            # Classe abstrata e utilitários de cálculo
│   ├── reward_single.py          # Recompensa PBRS para jogador único
│   └── reward_multi.py           # Recompensa cooperativa multi-agente
├── rsoccer_gym/
│   ├── ssl/
│   │   ├── ssl_el.py             # Ambiente SSL Entry Level e Division B
│   │   ├── ssl_el_multi.py       # Ambiente multi-agente cooperativo
│   │   └── ssl_gym_base.py       # Classe base de comunicação com simulador
│   ├── vss/                      # Ambientes legados IEEE VSSS
│   └── Simulators/               # Wrapper de comunicação com rSim (C++)
├── mappo/
│   ├── buffer.py                 # Buffer de trajetórias para múltiplos agentes
│   ├── models.py                 # Atores contínuos e Crítico Centralizado
│   └── mappo.py                  # Algoritmo MAPPO e rotina de treinamento
├── modelos/                      # Modelos treinados (.zip, .pt, .onnx, .json)
├── logs/
│   ├── checkpoints/              # Checkpoints monitorados pelo play_live.py
│   └── tensorboard/              # Telemetria e curvas de aprendizado
├── tests/                        # Suíte de testes unitários e de regressão
├── train.py                      # Treinador universal controlado via YAML
├── play.py                       # Visualizador de modelos salvos
├── play_live.py                  # Monitor em tempo real com auto-reload
├── export.py                     # Exportador ONNX e metadados para ROS 2
├── pyproject.toml                # Metadados do pacote e configurações de ferramentas
└── README.md                     # Documentação oficial do projeto
```

---

## Referências e Créditos

Este repositório é desenvolvido e mantido pela equipe **OxeBots** (Universidade Federal da Bahia - UFBA) para competições de futebol de robôs da RoboCup e IEEE.

A base de simulação utiliza componentes derivados do projeto original **rSoccer**:

```bibtex
@InProceedings{10.1007/978-3-030-98682-7_14,
  author    = {Martins, Felipe B. and Machado, Mateus G. and Bassani, Hansenclever F. and Braga, Pedro H. M. and Barros, Edna S.},
  editor    = {Alami, Rachid and Biswas, Joydeep and Cakmak, Maya and Obst, Oliver},
  title     = {rSoccer: A Framework for Studying Reinforcement Learning in Small and Very Small Size Robot Soccer},
  booktitle = {RoboCup 2021: Robot World Cup XXIV},
  year      = {2022},
  publisher = {Springer International Publishing},
  pages     = {165--176},
  isbn      = {978-3-030-98682-7}
}
```

---

## Licença

Distribuído sob a licença MIT. Consulte o arquivo [LICENSE](LICENSE) para mais detalhes.
