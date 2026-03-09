from numpy import ndarray, zeros, polynomial, array, outer

from src.datamodels.models import GlobalData, Grid, Element
from src.jacobian.elem_univ import ElemUniv

def C_matrix_for_element(element: Element, elem_univ : ElemUniv, specific_heat: float, density: float, n: int = 2) -> ndarray:
    C_for_element = zeros((4, 4))

    for i in range(n * n):
        j = element.jacobians[i].J
        j_inv = element.jacobians[i].J_inv
        j_det = element.jacobians[i].J_det
        w = elem_univ.W[i]

        N = elem_univ.N[i]

        C_for_point = specific_heat * density * j_det * w * outer(N, N)
        C_for_element += C_for_point

    return C_for_element

def C_matrix_for_all(grid: Grid, elem_univ: ElemUniv, global_data: GlobalData, n: int = 2) -> None:
    for element in grid.elements:
        element.C = C_matrix_for_element(element, elem_univ, global_data.specific_heat, global_data.density, n)

def print_C_matrix(C: ndarray) -> None:
    print(f"C:\n{C}\n")

def print_C_for_all(grid: Grid) -> None:
    for i, element in enumerate(grid.elements):
        print(f"Element {i + 1}:")
        print_C_matrix(element.C)