# Guia de Treinamento: Atacante SSL-EL (Entry Level) 3v3

Este documento explica a estrutura, regras, função de recompensa e como treinar o **Atacante SSL-EL** sem driblador.

---

## 1. Onde fica cada arquivo / módulo?

* **Ambiente Gym e Função de Recompensa**:
  [`rsoccer_gym/ssl/ssl_el_attacker.py`](file:///home/matheus/rSoccer/rsoccer_gym/ssl/ssl_el_attacker.py)
  * **Classe**: `SSLELAttackerEnv` (registrada como `"SSL-EL-v0"`)
  * **Função de Recompensa**: Método `_calculate_reward_and_done(self)`
  * **Renderizador 2D**: Classe `SSLELRenderField`
* **Registro de Ambientes**:
  [`rsoccer_gym/__init__.py`](file:///home/matheus/rSoccer/rsoccer_gym/__init__.py)
* **Script de Treino do Atacante**:
  [`train_ssl_el_attacker.py`](file:///home/matheus/rSoccer/train_ssl_el_attacker.py)
* **Script de Visualização / Teste com Pygame**:
  [`play_ssl_el_attacker.py`](file:///home/matheus/rSoccer/play_ssl_el_attacker.py)

---

## 2. Configurações e Regras da SSL-EL (Entry Level)

1. **Dimensões do Campo**:
   * Comprimento: **4.5 metros** ($x \in [-2.25, +2.25]$)
   * Largura: **3.0 metros** ($y \in [-1.50, +1.50]$)
2. **Área de Pênalti / Goleiro**:
   * Comprimento no eixo Y (`penalty_width`): **1.350 metros** ($y \in [-0.675, +0.675]$)
   * Largura no eixo X (`penalty_length`): **0.50 metros** (estende-se 0.50m para dentro a partir da linha de fundo)
3. **Regra de Invasão de Área**:
   * 🛑 **O atacante NÃO PODE entrar na área adversária nem na própria área**. Se entrar, recebe penalidade severa ($-5.0$) e o episódio é encerrado imediatamente.
   * O atacante é forçado a chutar **de fora da área**!
3. **Velocidades e Atuadores**:
   * Velocidade linear máxima do robô (`max_v`): **1.5 m/s**
   * Velocidade máxima do chute frontal (`kick_speed_x`): **3.0 m/s**
   * Velocidade angular máxima (`max_w`): **5.0 rad/s**

4. **Driblador (Dribbler)**:
   * **DESABILITADO** (`dribbler = False`). O robô deve usar sua cinemática omnidirecional para se alinhar atrás da bola e disparar o chutador frontal (`kick_v_x`).

---

## 3. Estrutura da Função de Recompensa (Kick-Centric Differential Shaping)

A função de recompensa está localizada no método `_calculate_reward_and_done(self)` em [`rsoccer_gym/ssl/ssl_el_attacker.py`](file:///home/matheus/rSoccer/rsoccer_gym/ssl/ssl_el_attacker.py):

| Componente | Tipo | Valor | Motivação |
| :--- | :---: | :---: | :--- |
| **Gol Marcado** | Terminal | **$+40.0$ a $+50.0$** | Super recompensa por acertar o gol adversário (bônus para gol de chute potente). |
| **Invasão de Área** | Terminal | **$-5.0$** | Penalidade por invadir a área de 1.35m x 0.50m (falta grave). |
| **Gol Sofrido / Contra** | Terminal | **$-15.0$** | Penalidade caso a bola entre no próprio gol. |
| **Bola Fora** | Terminal | **$-2.0$** | Penalidade por chutar para fora do campo. |
| **Aproximação e Contorno** | $\Delta \Phi$ Diferencial | até $\pm 1.0$ | Potencial dinâmico: alvo a 9.5cm atrás da bola (contato) ou contorno lateral se ultrapassar a bola. |
| **Avanço da Bola ao Gol** | $\Delta \Phi$ Diferencial | até $\pm 2.0$ | Recompensa proporcional ao deslocamento da bola na direção da meta. |
| **Alinhamento com o Gol** | $\Delta \Phi$ Diferencial | até $\pm 0.5$ | Recompensa diferencial de orientação angular apontando para o gol. |
| **Disparo do Chute** | Contínuo/Impacto | $+2.0$ a $+4.0$ | Recompensa imediata ao acionar o chutador frontal posicionado na bola. |
| **Tiro em Alta Velocidade no Alvo** | Evento | $+6.0$ | Bônus para qualquer disparo veloz na direção do gol. |
| **Sensor Infravermelho** | Contínuo | $+0.05$ | Bônus quando a bola está perfeitamente alojada na cavidade do chutador. |
| **Repulsão da Área Adversária** | Contínuo | $< 0$ | Barreira repulsiva a 20cm da área para ensinar o robô a chutar de longe e desacelerar. |
| **Tempo e Energia** | Contínuo | $-0.005$ / step | Penalidade suave que força o robô a resolver a jogada rapidamente. |

---

## 4. Como Executar

### Ativar o Ambiente Virtual:
```bash
source venv/bin/activate
```

### Iniciar o Treinamento do Atacante:
```bash
python train_ssl_el_attacker.py
```
* Os checkpoints serão salvos automaticamente na pasta `./modelos/checkpoints_ssl_el_attacker/` a cada 200.000 passos.
* O modelo final será salvo em `./modelos/ssl_el_attacker_ppo_final.zip`.

### Monitorar Métricas com o TensorBoard:
Em outro terminal:
```bash
source venv/bin/activate
tensorboard --logdir ./tensorboard_ssl_el_attacker/
```
Abra no navegador em `http://localhost:6006`.

### Assistir ao Treino em Tempo Real (Durante a Execução):
Em outro terminal enquanto o treino roda:
```bash
python play_ssl_el_attacker.py
```
Abre a janela gráfica e recarrega automaticamente os novos checkpoints salvos na pasta `modelos/checkpoints_ssl_el_attacker/` conforme o treinamento avança!

### Executar e Testar Modelos Treinados (Player Universal):
```bash
python play.py
# Ou especificando o ambiente/modelo salvo na pasta modelos:
python play.py --env SSL-EL-v0 --model ssl_el_attacker_ppo_final
```

---

## 5. Como Modificar / Customizar Recompensas

Para ajustar os pesos da função de recompensa, abra o arquivo [`rsoccer_gym/ssl/ssl_el_attacker.py`](file:///home/matheus/rSoccer/rsoccer_gym/ssl/ssl_el_attacker.py) e altere os valores dentro de `_calculate_reward_and_done()`:

* **Aumentar o valor do Gol**: Modifique `reward = 15.0` para um valor maior (ex: `25.0`).
* **Tornar a punição de invasão mais rigorosa**: Modifique `reward = -5.0` para `reward = -10.0`.
* **Ajustar distância de alinhamento**: Altere `ideal_robot_pos = ball_pos - (vec_ball_goal * 0.20)`.
