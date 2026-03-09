from dataclasses import dataclass, field

from numpy import ndarray, array, zeros, ix_, printoptions, newaxis, linalg, full, amin, amax

from src.datamodels.models import Grid, GlobalData

@dataclass
class EquasionSystem:
    H_final: ndarray = field(init=False)
    P_final: ndarray = field(init=False)
    C_final: ndarray = field(init=False)

    T_static: ndarray = field(init=False)
    T_dynamic: ndarray = field(init=False)

    def __init__(self, global_data: GlobalData):
        n = global_data.n_nodes
        self.H_final = zeros((n, n))
        self.P_final = zeros(n)
        self.C_final = zeros((n, n))
        self.T_static = zeros(n)
        self.T_dynamic = zeros(n)

def aggregate_H(grid: Grid, matrixes: EquasionSystem) -> None:
    for element in grid.elements:
        id = array(element.node_ids) - 1
        local = element.H + element.Hbc
        idx, idy = ix_(id, id)
        matrixes.H_final[idx, idy] += local

def print_global_H(matrixes: EquasionSystem) -> None:
    with printoptions(linewidth=1000, suppress=True):
        print(matrixes.H_final)

def aggregate_P(grid: Grid, matrixes: EquasionSystem) -> None:
    for element in grid.elements:
        id = array(element.node_ids) - 1
        matrixes.P_final[id] += element.P

def print_global_P(matrixes: EquasionSystem) -> None:
    print(matrixes.P_final[:, newaxis])

def aggregate_C(grid: Grid, matrixes: EquasionSystem) -> None:
    for element in grid.elements:
        id = array(element.node_ids) - 1
        idx, idy = ix_(id, id)
        matrixes.C_final[idx, idy] += element.C

def print_global_C(matrixes: EquasionSystem) -> None:
    with printoptions(linewidth=1000, suppress=True):
        print(matrixes.C_final)

def print_T_static(matrixes: EquasionSystem) -> None:
    print(matrixes.T_static[:, newaxis])

def T_static_vector(matrixes: EquasionSystem) -> None:
    matrixes.T_static = linalg.solve(matrixes.H_final, matrixes.P_final)

def print_T_dynamic(matrixes: EquasionSystem) -> None:
    print(matrixes.T_dynamic[:, newaxis])

def print_T_dynamic_min_max(matrixes: EquasionSystem) -> None:
    T_min = amin(matrixes.T_dynamic)
    T_max = amax(matrixes.T_dynamic)
    print(f"MIN: {T_min}")
    print(f"MAX: {T_max}")

def T_dynamic_vector(matrixes: EquasionSystem, SimulationStepTime: float, T0: ndarray) -> None:
    H_sub = matrixes.H_final + (matrixes.C_final / SimulationStepTime) # macierz
    P_sub = matrixes.P_final + ((matrixes.C_final / SimulationStepTime) @ T0) # wektor

    matrixes.T_dynamic = linalg.solve(H_sub, P_sub)

def T_dynamic_simulation(global_data: GlobalData, matrixes: EquasionSystem) -> None:
    T0 = full(global_data.n_nodes, global_data.initial_temp, dtype=float)
    end = int(global_data.simulation_time)
    step = int(global_data.simulation_step_time)

    for i in range(step, end + 1, step):
        T_dynamic_vector(matrixes, global_data.simulation_step_time, T0)
        T0 = matrixes.T_dynamic
        print(f"Time: {i}")
        print_T_dynamic(matrixes)

def T_dynamic_min_max(global_data: GlobalData, matrixes: EquasionSystem) -> None:
    T0 = full(global_data.n_nodes, global_data.initial_temp, dtype=float)
    end = int(global_data.simulation_time)
    step = int(global_data.simulation_step_time)

    for i in range(step, end + 1, step):
        T_dynamic_vector(matrixes, global_data.simulation_step_time, T0)
        T0 = matrixes.T_dynamic
        print(f"Time: {i}")
        print_T_dynamic_min_max(matrixes)