import os

import numpy as np
import torch
from torch import nn, optim

from mappo.buffer import MultiAgentRolloutBuffer
from mappo.models import ActorONNXWrapper, CentralizedCritic, ContinuousActor


class MAPPOTrainer:
    """
    Treinador MAPPO (Multi-Agent PPO) com CTDE (Centralized Training, Decentralized Execution):
    - Ator compartilhado (com identificadores One-Hot) ou atores individuais.
    - Crítico Centralizado observando o estado global s.
    - PPO Clipped Objective com Normalização de Vantagem.
    - Value Clipping e Huber Loss para estabilidade.
    - Suporte a salvamento em .pt e exportação simultânea em .onnx, com suporte a continuação de treino (resume).
    """

    def __init__(
        self,
        agents: list[str],
        obs_dim: int = 35,
        state_dim: int = 53,
        act_dim: int = 4,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        clip_param: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.001,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 5,
        batch_size: int = 128,
        use_shared_actor: bool = True,
        device: str = "cpu",
    ):
        self.agents = agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.use_shared_actor = use_shared_actor
        self.device = device

        # Redes Neurais
        if use_shared_actor:
            self.actor = ContinuousActor(obs_dim, act_dim).to(device)
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor, eps=1e-5)
        else:
            self.actors = {a: ContinuousActor(obs_dim, act_dim).to(device) for a in agents}
            self.actor_optimizers = {
                a: optim.Adam(self.actors[a].parameters(), lr=lr_actor, eps=1e-5) for a in agents
            }

        self.critic = CentralizedCritic(state_dim).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic, eps=1e-5)

    def get_actions(self, obs_dict: dict[str, np.ndarray], deterministic: bool = False):
        """Coleta ações descentralizadas para cada agente."""
        actions_dict = {}
        log_probs_dict = {}

        with torch.no_grad():
            for a in self.agents:
                obs_t = torch.as_tensor(obs_dict[a], device=self.device, dtype=torch.float32)
                if self.use_shared_actor:
                    act, log_p, _ = self.actor.get_action(obs_t, deterministic=deterministic)
                else:
                    act, log_p, _ = self.actors[a].get_action(obs_t, deterministic=deterministic)
                actions_dict[a] = act
                log_probs_dict[a] = log_p

        return actions_dict, log_probs_dict

    def get_value(self, state: np.ndarray):
        """Avalia o valor do estado global com o Crítico Centralizado."""
        with torch.no_grad():
            state_t = torch.as_tensor(state, device=self.device, dtype=torch.float32)
            value = self.critic(state_t)
        return value

    def train_step(self, buffer: MultiAgentRolloutBuffer) -> dict[str, float]:
        """Executa a atualização de parâmetros da política e do crítico centralizado."""
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        update_count = 0

        for _ in range(self.ppo_epochs):
            data_generator = buffer.get_generator(self.batch_size)

            for batch in data_generator:
                (
                    b_obs,
                    b_states,
                    b_actions,
                    b_log_probs,
                    b_values,
                    b_returns,
                    b_advs,
                ) = batch

                # 1. Otimização do Crítico Centralizado
                current_values = self.critic(b_states)
                v_clipped = b_values + (current_values - b_values).clamp(-self.clip_param, self.clip_param)
                vf_loss_1 = (current_values - b_returns).pow(2)
                vf_loss_2 = (v_clipped - b_returns).pow(2)
                critic_loss = 0.5 * torch.max(vf_loss_1, vf_loss_2).mean()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                # 2. Otimização do Ator (ou Atores)
                if self.use_shared_actor:
                    actor_loss_sum = 0.0
                    ent_sum = 0.0
                    for a in self.agents:
                        new_log_prob, ent = self.actor.evaluate_actions(b_obs[a], b_actions[a])
                        ratio = torch.exp(new_log_prob - b_log_probs[a])
                        surr1 = ratio * b_advs
                        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * b_advs
                        actor_loss = -torch.min(surr1, surr2).mean()
                        actor_loss_sum += actor_loss - self.entropy_coef * ent.mean()
                        ent_sum += ent.mean().item()

                    self.actor_optimizer.zero_grad()
                    actor_loss_sum.backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.actor_optimizer.step()

                    total_actor_loss += actor_loss_sum.item()
                    total_entropy += ent_sum / len(self.agents)
                else:
                    for a in self.agents:
                        new_log_prob, ent = self.actors[a].evaluate_actions(b_obs[a], b_actions[a])
                        ratio = torch.exp(new_log_prob - b_log_probs[a])
                        surr1 = ratio * b_advs
                        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * b_advs
                        actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * ent.mean()

                        self.actor_optimizers[a].zero_grad()
                        actor_loss.backward()
                        nn.utils.clip_grad_norm_(self.actors[a].parameters(), self.max_grad_norm)
                        self.actor_optimizers[a].step()

                        total_actor_loss += actor_loss.item()
                        total_entropy += ent.mean().item()

                total_critic_loss += critic_loss.item()
                update_count += 1

        return {
            "actor_loss": total_actor_loss / max(update_count, 1),
            "critic_loss": total_critic_loss / max(update_count, 1),
            "entropy": total_entropy / max(update_count, 1),
        }

    def export_onnx(self, onnx_filepath: str):
        """Exporta o ator treinado para formato ONNX para implantação rápida."""
        os.makedirs(os.path.dirname(onnx_filepath) or ".", exist_ok=True)
        dummy_input = torch.randn(1, self.obs_dim, dtype=torch.float32, device="cpu")

        if self.use_shared_actor:
            onnx_wrapper = ActorONNXWrapper(self.actor).to("cpu")
            onnx_wrapper.eval()
            with torch.no_grad():
                torch.onnx.export(
                    onnx_wrapper,
                    dummy_input,
                    onnx_filepath,
                    export_params=True,
                    opset_version=18,
                    do_constant_folding=True,
                    input_names=["observation"],
                    output_names=["action"],
                    dynamic_axes={"observation": {0: "batch_size"}, "action": {0: "batch_size"}},
                )
            self.actor.to(self.device)
        else:
            base, ext = os.path.splitext(onnx_filepath)
            for a in self.agents:
                agent_onnx = f"{base}_{a}{ext}"
                onnx_wrapper = ActorONNXWrapper(self.actors[a]).to("cpu")
                onnx_wrapper.eval()
                with torch.no_grad():
                    torch.onnx.export(
                        onnx_wrapper,
                        dummy_input,
                        agent_onnx,
                        export_params=True,
                        opset_version=18,
                        do_constant_folding=True,
                        input_names=["observation"],
                        output_names=["action"],
                        dynamic_axes={"observation": {0: "batch_size"}, "action": {0: "batch_size"}},
                    )
                self.actors[a].to(self.device)


    def save(self, filepath: str, extra_info: dict | None = None, export_onnx: bool = True):
        """
        Salva o checkpoint em formato PyTorch (.pt) com pesos e estado dos otimizadores
        e exporta simultaneamente o modelo em formato ONNX (.onnx).
        """
        if not filepath.endswith(".pt"):
            pt_path = filepath + ".pt"
        else:
            pt_path = filepath

        os.makedirs(os.path.dirname(pt_path) or ".", exist_ok=True)

        checkpoint = {
            "critic": self.critic.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "use_shared_actor": self.use_shared_actor,
            "obs_dim": self.obs_dim,
            "state_dim": self.state_dim,
            "act_dim": self.act_dim,
        }

        if self.use_shared_actor:
            checkpoint["actor"] = self.actor.state_dict()
            checkpoint["actor_optimizer"] = self.actor_optimizer.state_dict()
        else:
            checkpoint["actors"] = {a: self.actors[a].state_dict() for a in self.agents}
            checkpoint["actor_optimizers"] = {a: self.actor_optimizers[a].state_dict() for a in self.agents}

        if extra_info:
            checkpoint.update(extra_info)

        torch.save(checkpoint, pt_path)

        if export_onnx:
            onnx_path = pt_path[:-3] + ".onnx"
            try:
                self.export_onnx(onnx_path)
            except (RuntimeError, OSError, ValueError) as e:
                print(f"[!] Aviso: Falha ao exportar ONNX ({e})")

    def load(self, filepath: str, load_optimizers: bool = True) -> dict:
        """
        Carrega os pesos do modelo e otimizadores para execução ou continuação de treino (resume).
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        # Restaurar Crítico
        self.critic.load_state_dict(checkpoint["critic"])
        if load_optimizers and "critic_optimizer" in checkpoint:
            try:
                self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
            except (RuntimeError, KeyError):
                pass

        # Restaurar Ator
        if checkpoint.get("use_shared_actor", True):
            self.actor.load_state_dict(checkpoint["actor"])
            if load_optimizers and "actor_optimizer" in checkpoint:
                try:
                    self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
                except (RuntimeError, KeyError):
                    pass
        else:
            for a in self.agents:
                self.actors[a].load_state_dict(checkpoint["actors"][a])
                if load_optimizers and "actor_optimizers" in checkpoint:
                    try:
                        self.actor_optimizers[a].load_state_dict(checkpoint["actor_optimizers"][a])
                    except (RuntimeError, KeyError):
                        pass

        return checkpoint


def train_mappo(config: dict):
    """Pipeline de treinamento Multi-Agent (MAPPO com CTDE)."""
    import gymnasium as gym

    import rsoccer_gym  # noqa: F401

    env_cfg = config.get("environment", {})
    mon_cfg = config.get("monitoring", {})
    train_cfg = config.get("training", {})
    mappo_cfg = config.get("mappo", {})

    field_params = env_cfg.get("field", {})
    env = gym.make("SSL-EL-Coop-v0", field_params=field_params)

    agents = env.unwrapped.agents
    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = MAPPOTrainer(
        agents=agents,
        obs_dim=35,
        state_dim=53,
        act_dim=4,
        lr_actor=mappo_cfg.get("lr_actor", 3e-4),
        lr_critic=mappo_cfg.get("lr_critic", 1e-3),
        clip_param=mappo_cfg.get("clip_param", 0.2),
        value_loss_coef=mappo_cfg.get("value_loss_coef", 0.5),
        entropy_coef=mappo_cfg.get("entropy_coef", 0.001),
        max_grad_norm=mappo_cfg.get("max_grad_norm", 0.5),
        ppo_epochs=mappo_cfg.get("ppo_epochs", 5),
        batch_size=mappo_cfg.get("batch_size", 128),
        use_shared_actor=mappo_cfg.get("use_shared_actor", True),
        device=device,
    )

    num_steps = mappo_cfg.get("buffer_size", 2048)
    buffer = MultiAgentRolloutBuffer(
        num_steps=num_steps,
        num_envs=1,
        agents=agents,
        obs_dim=35,
        state_dim=53,
        act_dim=4,
        gamma=mappo_cfg.get("gamma", 0.99),
        gae_lambda=mappo_cfg.get("gae_lambda", 0.95),
        device=device,
    )

    total_timesteps = train_cfg.get("total_timesteps", 1_000_000)
    checkpoints_dir = mon_cfg.get("checkpoints_dir", "./logs/checkpoints_multi/")
    final_model_path = train_cfg.get("final_model_path", "./modelos/ssl_el_coop_mappo_final")

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(os.path.dirname(final_model_path) or ".", exist_ok=True)

    print(f"[INFO] MAPPO configurado para {len(agents)} agentes no dispositivo: {device.upper()}")
    print(f"[INFO] Treinamento configurado para {total_timesteps:,} passos.")

    obs_dict, _ = env.reset(seed=train_cfg.get("seed", 42))
    step_count = 0
    save_freq = train_cfg.get("checkpoint_freq", 50000)

    try:
        while step_count < total_timesteps:
            for _ in range(num_steps):
                state = env.unwrapped._get_global_state() if hasattr(env.unwrapped, "_get_global_state") else np.zeros(53, dtype=np.float32)
                actions_dict, log_probs_dict = trainer.get_actions(obs_dict, deterministic=False)
                value = trainer.get_value(state)

                np_actions = {a: actions_dict[a].cpu().numpy().squeeze(0) for a in agents}
                next_obs, rewards, dones, truncs, _ = env.step(np_actions)
                step_count += 1

                buffer.insert(
                    obs_dict=obs_dict,
                    state=state,
                    actions_dict=actions_dict,
                    log_probs_dict=log_probs_dict,
                    values=value,
                    rewards_dict=rewards,
                    dones_dict=dones,
                )

                obs_dict = next_obs
                done_any = any(dones.values()) if isinstance(dones, dict) else bool(dones)
                trunc_any = any(truncs.values()) if isinstance(truncs, dict) else bool(truncs)

                if done_any or trunc_any:
                    obs_dict, _ = env.reset()

                if step_count >= total_timesteps:
                    break

            final_state = env.unwrapped._get_global_state() if hasattr(env.unwrapped, "_get_global_state") else np.zeros(53, dtype=np.float32)
            next_val = trainer.get_value(final_state)
            buffer.compute_gae(next_value=next_val, next_done=torch.zeros((1, 1), device=device))
            trainer.train_step(buffer)

            if step_count % save_freq == 0:
                ckpt_path = os.path.join(checkpoints_dir, f"mappo_step_{step_count}.pt")
                trainer.save(ckpt_path)
                print(f"[OK] Checkpoint MAPPO salvo em: {ckpt_path} (Passo {step_count:,})")

        trainer.save(final_model_path)
        print("\n[OK] Treinamento MAPPO concluído com sucesso!")
        print(f"[INFO] Modelo final salvo em: {final_model_path}.pt / .onnx")
    finally:
        env.close()
