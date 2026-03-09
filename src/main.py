import os.path

from src.datamodels.data_loader import parse_mesh_file
from src.jacobian.elem_univ import build_elem_univ
from src.jacobian.J_jacobian import jacobian_for_all, print_jacobian_for_all
from src.globalequations.H_matrix import H_matrix_for_all, print_H_for_all
from src.globalequations.Hbc_matrix import Hbc_matrix_for_all, print_Hbc_for_all
from src.globalequations.P_vector import P_vector_for_all, print_P_for_all
from src.globalequations.C_matrix import C_matrix_for_all, print_C_for_all
from src.globalequations.equations import EquasionSystem, aggregate_H, print_global_H, aggregate_P, print_global_P, print_T_static, T_static_vector, print_T_dynamic, aggregate_C, print_global_C, T_dynamic_vector, T_dynamic_simulation, T_dynamic_min_max

base = r"C:\Users\Mateusz\Documents\S5\MES2\Siatki MES" # hardkodowane bo inaczej nie działą
filename = ["Test1_4_4.txt", "Test2_4_4_MixGrid.txt", "Test3_31_31_kwadrat.txt"]

n = 4
i = 1

def main():
    # (1) Wczytywanie
    filepath = os.path.join(base, filename[i])
    global_data, grid = parse_mesh_file(filepath)

    # Test Wczytywania
    #print(global_data.summary())
    #print(grid.summary())

    # (3) Jakobian
    elem_univ = build_elem_univ(n)
    jacobian_for_all(grid, elem_univ, n)

    # Test Jakobianu
    #print_jacobian_for_all(grid)

    # (4) Macierz H
    H_matrix_for_all(grid, elem_univ, global_data, n)

    # Test macierzy H
    #print_H_for_all(grid)

    # (6) Macierz Hbc
    Hbc_matrix_for_all(grid, global_data, n)

    #Test macierzy Hbc
    #print_Hbc_for_all(grid)

    # (5) Agregacja macierzy H
    matrixes = EquasionSystem(global_data)
    aggregate_H(grid, matrixes)

    # Test agregacji macierzy H
    #print_global_H(matrixes)

    # (7) Wektor P
    P_vector_for_all(grid, global_data, n)
    aggregate_P(grid, matrixes)

    # Test wektora P
    #print_P_for_all(grid)
    #print_global_P(matrixes)

    # (8) Macierz C
    C_matrix_for_all(grid, elem_univ, global_data, n)
    aggregate_C(grid, matrixes)

    # Test macierzy C
    #print_C_for_all(grid)
    #print_global_C(matrixes)

    # (2) Metoda Gaussa
    #T_static_vector(matrixes)

    # Test metody Gaussa
    #print_T_static(matrixes)

    # (9) Symulacja

    # Test symulacji
    #T_dynamic_simulation(global_data, matrixes)

    # (10) Sprawozdanie

    # Test sprawozdania
    T_dynamic_min_max(global_data, matrixes)

if __name__ == "__main__":
    main()