from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class Location:
    """Класс, представляющий местоположение (поставщик или потребитель)"""

    name: str
    capacity: int
    demand: int


@dataclass
class TransportationProblem:
    """Класс, представляющий транспортную задачу"""

    suppliers: List[Location]
    consumers: List[Location]
    costs: np.ndarray
    solution: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        total_supply: int = sum(s.capacity for s in self.suppliers)
        total_demand: int = sum(c.demand for c in self.consumers)
        if total_supply != total_demand:
            raise ValueError("Общее предложение должно равняться общему спросу")

        if self.solution is None:
            self.solution = np.zeros((len(self.suppliers), len(self.consumers)))

    def get_total_cost(self) -> float:
        """Вычисляет общую стоимость транспортировки на основе текущего решения"""
        if self.solution is None:
            return 0.0
        return float(np.sum(self.solution * self.costs))

    def is_balanced(self) -> bool:
        """Проверяет, сбалансирована ли задача (общее предложение = общий спрос)"""
        total_supply: int = sum(s.capacity for s in self.suppliers)
        total_demand: int = sum(c.demand for c in self.consumers)
        return total_supply == total_demand
