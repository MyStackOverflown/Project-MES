import os.path

from data_loader import parse_mesh_file
from elem_univ import build_elem_univ
from J_jacobian import jacobian_for_all, print_jacobian_for_all
from H_matrix import H_matrix_for_all, print_H_for_all

base = r"C:\Users\Mateusz\Documents\S5\MES2\Siatki MES"
filename = ["Test1_4_4.txt", "Test2_4_4_MixGrid.txt", "Test3_31_31_kwadrat.txt"]

n = 2

def main():
    # (1) Wczytywanie
    filepath = os.path.join(base, filename[1])
    global_data, grid = parse_mesh_file(filepath)

    # Test Wczytywania
    # print(global_data.summary())
    # print(grid.summary())

    # (2) Jakobian
    elem_univ = build_elem_univ(n)
    jacobian_for_all(grid, elem_univ)

    # Test Jakobianu
    print_jacobian_for_all(grid)

    # (3) Macierz H
    H_matrix_for_all(grid, elem_univ, global_data)

    # Test macierzy H
    print_H_for_all(grid)

if __name__ == "__main__":
    main()