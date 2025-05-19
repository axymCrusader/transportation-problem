from typing import Dict

import numpy as np

from src.models import Location, TransportationProblem
from src.solvers import (
    LeastCostSolver,
    NorthwestCornerSolver,
    TransportationSolver,
    VogelsApproximationSolver,
)


def print_input_data(problem: TransportationProblem) -> None:
    """Выводит исходные данные задачи в табличном формате"""
    print("\nИсходные данные задачи:")
    print("\nМатрица стоимостей перевозок (руб.):")
    print(" " * 15, end="")
    for consumer in problem.consumers:
        print(f"{consumer.name:>15}", end="")
    print("\n")

    for i, supplier in enumerate(problem.suppliers):
        print(f"{supplier.name:15}", end="")
        for j in range(len(problem.consumers)):
            print(f"{problem.costs[i, j]:15.0f}", end="")
        print(f" | {supplier.capacity:5.0f} (предложение)")

    print("-" * 60)
    print("Спрос:", end="")
    for consumer in problem.consumers:
        print(f"{consumer.demand:15.0f}", end="")
    print("\n")

    print(
        f"Общий объем предложения: {sum(s.capacity for s in problem.suppliers):,.0f} ед."
    )
    print(f"Общий объем спроса: {sum(c.demand for c in problem.consumers):,.0f} ед.")
    print("\n" + "=" * 60 + "\n")


def create_problem() -> TransportationProblem:
    """Создает экземпляр транспортной задачи с тестовыми данными"""
    suppliers: list[Location] = [
        Location("Омск", 1000, 0),
        Location("Новосибирск", 1700, 0),
        Location("Томск", 1600, 0),
    ]

    consumers: list[Location] = [
        Location("Нижний Новгород", 0, 1600),
        Location("Пермь", 0, 1000),
        Location("Краснодар", 0, 1700),
    ]

    costs: np.ndarray = np.array(
        [
            [1000, 1200, 2000],  # Омск
            [1500, 1300, 1800],  # Новосибирск
            [1400, 1100, 1900],  # Томск
        ]
    )

    return TransportationProblem(suppliers, consumers, costs)


def print_solution(problem: TransportationProblem, method_name: str) -> None:
    """Выводит решение задачи в табличном формате"""
    if problem.solution is None:
        raise ValueError(
            "Сначала вызовите solver.solve(problem), чтобы получить решение"
        )

    print(f"\nРешение методом {method_name}:")
    print("\nМатрица поставок:")
    print(" " * 15, end="")
    for consumer in problem.consumers:
        print(f"{consumer.name:>15}", end="")
    print("\n")

    solution: np.ndarray = problem.solution

    for i, supplier in enumerate(problem.suppliers):
        print(f"{supplier.name:15}", end="")
        for j in range(len(problem.consumers)):
            print(f"{solution[i, j]:15.0f}", end="")
        print(f" | {supplier.capacity:5.0f}")

    print("-" * 60)
    print("Спрос:", end="")
    for consumer in problem.consumers:
        print(f"{consumer.demand:15.0f}", end="")
    print("\n")

    print(f"Общая стоимость перевозки: {problem.get_total_cost():,.2f} руб.")


def main() -> None:
    problem: TransportationProblem = create_problem()
    print_input_data(problem)

    northwest_solver: TransportationSolver = NorthwestCornerSolver()
    least_cost_solver: TransportationSolver = LeastCostSolver()
    vogel_solver: TransportationSolver = VogelsApproximationSolver()

    solvers: Dict[str, TransportationSolver] = {
        "Северо-западного угла": northwest_solver,
        "Наименьшей стоимости": least_cost_solver,
        "Фогеля": vogel_solver,
    }

    for method_name, solver in solvers.items():
        current_problem = TransportationProblem(
            problem.suppliers.copy(), problem.consumers.copy(), problem.costs.copy()
        )

        try:
            solver.solve(current_problem)
            print_solution(current_problem, method_name)
        except Exception as e:
            print(f"Error solving with {method_name}: {e}")


if __name__ == "__main__":
    main()
