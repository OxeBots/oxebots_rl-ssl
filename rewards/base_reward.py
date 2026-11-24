from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Optional


class BaseReward(ABC):
    """
    Classe base abstrata para todas as funções de recompensa do OxeBots RL-SSL.
    Desacopla a simulação física da lógica de aprendizado por reforço.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.rewards_cfg = self.config.get("rewards", {})
        self.penalties_cfg = self.config.get("penalties", {})

    def get_reward_val(self, name: str, default: float = 0.0) -> float:
        """Retorna o valor do bônus se enabled=true, caso contrário 0.0."""
        item = self.rewards_cfg.get(name, {})
        if isinstance(item, dict):
            return float(item.get("value", default)) if item.get("enabled", True) else 0.0
        return float(item) if item else 0.0

    def get_penalty_val(self, name: str, default: float = 0.0) -> float:
        """Retorna o valor da penalidade se enabled=true, caso contrário 0.0."""
        item = self.penalties_cfg.get(name, {})
        if isinstance(item, dict):
            return float(item.get("value", default)) if item.get("enabled", True) else 0.0
        return float(item) if item else 0.0

    def is_enabled(self, category: str, name: str, default: bool = True) -> bool:
        """Verifica se determinado componente de recompensa/penalidade está ativado."""
        section = self.rewards_cfg if category == "rewards" else self.penalties_cfg
        item = section.get(name, {})
        if isinstance(item, dict):
            return bool(item.get("enabled", default))
        return bool(item)

    @abstractmethod
    def calculate(
        self, env, action, last_frame, frame, sent_commands
    ) -> Tuple[float, bool, Dict[str, float]]:
        """
        Calcula a recompensa e condição de término para o passo atual.
        
        Retorna:
            reward (float): Valor numérico total da recompensa no passo.
            done (bool): Se o episódio terminou (gol, falta, saída).
            info (dict): Dicionário detalhado com cada parcela para o TensorBoard.
        """
        pass

    def reset(self):
        """Reseta variáveis internas de potencial e rastreamento a cada episódio."""
        pass
