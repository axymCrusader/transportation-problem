from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np
from .models import TransportationProblem


class TransportationSolver(ABC):
    """Абстрактный базовый класс для решателей транспортной задачи"""

    @abstractmethod
    def solve(self, problem: TransportationProblem) -> np.ndarray:
        """Решает транспортную задачу и возвращает матрицу решения"""
        pass

    def _validate_problem(self, problem: TransportationProblem) -> None:
        """Проверяет корректность данных задачи"""
        if not isinstance(problem, TransportationProblem):
            raise TypeError("Задача должна быть экземпляром TransportationProblem")
        if not problem.is_balanced():
            raise ValueError(
                "Задача должна быть сбалансирована (общее предложение = общий спрос)"
            )
        if problem.costs.shape != (len(problem.suppliers), len(problem.consumers)):
            raise ValueError(
                "Размерности матрицы стоимостей не соответствуют количеству поставщиков и потребителей"
            )


class NorthwestCornerSolver(TransportationSolver):
    """Реализация метода северо-западного угла"""

    def solve(self, problem: TransportationProblem) -> np.ndarray:
        self._validate_problem(problem)

        solution: np.ndarray = np.zeros(
            (len(problem.suppliers), len(problem.consumers))
        )
        supply: np.ndarray = np.array(
            [s.capacity for s in problem.suppliers], dtype=int
        )
        demand: np.ndarray = np.array([c.demand for c in problem.consumers], dtype=int)

        i: int = 0
        j: int = 0
        max_iterations: int = len(supply) * len(demand)
        iteration: int = 0

        while i < len(supply) and j < len(demand) and iteration < max_iterations:
            if supply[i] == 0:
                i += 1
                continue
            if demand[j] == 0:
                j += 1
                continue

            amount: int = min(supply[i], demand[j])
            solution[i, j] = amount
            supply[i] -= amount
            demand[j] -= amount

            iteration += 1

        if iteration >= max_iterations:
            raise RuntimeError(
                "Превышено максимальное количество итераций в методе северо-западного угла"
            )

        problem.solution = solution
        return solution


class LeastCostSolver(TransportationSolver):
    """Реализация метода наименьшей стоимости"""

    def solve(self, problem: TransportationProblem) -> np.ndarray:
        self._validate_problem(problem)

        solution: np.ndarray = np.zeros(
            (len(problem.suppliers), len(problem.consumers))
        )
        supply: np.ndarray = np.array(
            [s.capacity for s in problem.suppliers], dtype=int
        )
        demand: np.ndarray = np.array([c.demand for c in problem.consumers], dtype=int)
        costs: np.ndarray = problem.costs.astype(float)

        max_iterations: int = len(supply) * len(demand)
        iteration: int = 0

        while np.any(supply > 0) and np.any(demand > 0) and iteration < max_iterations:
            min_cost: float = np.inf
            min_pos: Optional[Tuple[int, int]] = None

            for i in range(len(supply)):
                if supply[i] <= 0:
                    continue
                for j in range(len(demand)):
                    if demand[j] <= 0:
                        continue
                    if costs[i, j] < min_cost:
                        min_cost = costs[i, j]
                        min_pos = (i, j)

            if min_pos is None:
                break

            i, j = min_pos
            amount: int = min(supply[i], demand[j])
            solution[i, j] = amount
            supply[i] -= amount
            demand[j] -= amount

            if supply[i] == 0:
                costs[i, :] = np.inf
            if demand[j] == 0:
                costs[:, j] = np.inf

            iteration += 1

        if iteration >= max_iterations:
            raise RuntimeError(
                "Превышено максимальное количество итераций в методе наименьшей стоимости"
            )

        problem.solution = solution
        return solution


class VogelsApproximationSolver(TransportationSolver):
    """Реализация метода аппроксимации Фогеля"""

    def _get_penalties(
        self, costs: np.ndarray, supply: np.ndarray, demand: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Вычисляет штрафы по строкам и столбцам"""
        row_penalties: np.ndarray = np.zeros(len(supply))
        col_penalties: np.ndarray = np.zeros(len(demand))

        for i in range(len(supply)):
            if supply[i] > 0:
                row_valid_costs: np.ndarray = costs[i, demand > 0]
                if len(row_valid_costs) >= 2:
                    row_sorted_costs: np.ndarray = np.sort(row_valid_costs)
                    row_penalties[i] = row_sorted_costs[1] - row_sorted_costs[0]

        for j in range(len(demand)):
            if demand[j] > 0:
                col_valid_costs: np.ndarray = costs[supply > 0, j]
                if len(col_valid_costs) >= 2:
                    col_sorted_costs: np.ndarray = np.sort(col_valid_costs)
                    col_penalties[j] = col_sorted_costs[1] - col_sorted_costs[0]

        return row_penalties, col_penalties

    def solve(self, problem: TransportationProblem) -> np.ndarray:
        self._validate_problem(problem)

        solution: np.ndarray = np.zeros(
            (len(problem.suppliers), len(problem.consumers))
        )
        supply: np.ndarray = np.array(
            [s.capacity for s in problem.suppliers], dtype=int
        )
        demand: np.ndarray = np.array([c.demand for c in problem.consumers], dtype=int)
        costs: np.ndarray = problem.costs.astype(float)

        total_supply: int = np.sum(supply)
        total_demand: int = np.sum(demand)
        remaining_supply: int = total_supply
        remaining_demand: int = total_demand

        max_iterations: int = len(supply) * len(demand) * 2
        iteration: int = 0

        while (
            remaining_supply > 0 and remaining_demand > 0 and iteration < max_iterations
        ):
            row_penalties, col_penalties = self._get_penalties(costs, supply, demand)

            if np.all(row_penalties == 0) and np.all(col_penalties == 0):
                remaining_valid_costs: np.ndarray = costs.copy()
                remaining_valid_costs[supply <= 0, :] = np.inf
                remaining_valid_costs[:, demand <= 0] = np.inf

                if np.all(np.isinf(remaining_valid_costs)):
                    break

                min_cost_pos: Tuple[int, int] = np.unravel_index(
                    np.argmin(remaining_valid_costs), remaining_valid_costs.shape
                )
                min_i, min_j = min_cost_pos
            else:
                max_row_penalty: float = np.max(row_penalties)
                max_col_penalty: float = np.max(col_penalties)

                if max_row_penalty >= max_col_penalty:
                    row_idx: int = np.argmax(row_penalties)
                    valid_demand: np.ndarray = demand > 0
                    if not np.any(valid_demand):
                        break
                    col_idx: int = np.argmin(costs[row_idx, valid_demand])
                    col_idx = np.where(valid_demand)[0][col_idx]
                    min_i, min_j = row_idx, col_idx
                else:
                    col_idx_alt: int = np.argmax(col_penalties)
                    valid_supply: np.ndarray = supply > 0
                    if not np.any(valid_supply):
                        break
                    row_idx_alt: int = np.argmin(costs[valid_supply, col_idx_alt])
                    row_idx_alt = np.where(valid_supply)[0][row_idx_alt]
                    min_i, min_j = row_idx_alt, col_idx_alt

            amount: int = min(supply[min_i], demand[min_j])
            solution[min_i, min_j] = amount
            supply[min_i] -= amount
            demand[min_j] -= amount

            remaining_supply -= amount
            remaining_demand -= amount

            if supply[min_i] == 0:
                costs[min_i, :] = np.inf
            if demand[min_j] == 0:
                costs[:, min_j] = np.inf

            iteration += 1

        if remaining_supply > 0 or remaining_demand > 0:
            raise RuntimeError(
                "Не удалось найти полное решение методом аппроксимации Фогеля"
            )

        problem.solution = solution
        return solution
