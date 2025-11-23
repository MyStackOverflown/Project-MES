from models import GlobalData, Grid, Element

from numpy import ndarray, zeros, array, outer

from elem_univ import ElemUniv


def H_matrix_for_element(element: Element, elem_univ : ElemUniv, conductivity: float) -> ndarray:
    H_for_element = zeros((4, 4))
    n = len(elem_univ.W)

    for i in range(n):
        j = element.jacobians[i].J
        j_inv = element.jacobians[i].J_inv
        j_det = element.jacobians[i].J_det
        w = elem_univ.W[i]

        local_d = array([elem_univ.dN_dXi[i], elem_univ.dN_dEta[i]])
        global_d = j_inv @ local_d

        dN_dx = global_d[0]
        dN_dy = global_d[1]

        X = outer(dN_dx, dN_dx)
        Y = outer(dN_dy, dN_dy)

        H_for_point = conductivity * (X + Y) * j_det * w

        H_for_element += H_for_point

    return H_for_element

def H_matrix_for_all(grid: Grid, elem_univ: ElemUniv, global_data: GlobalData) -> None:
    for element in grid.elements:
        element.H = H_matrix_for_element(element, elem_univ, global_data.conductivity)

def H_global_matrix():
    pass

def print_H_matrix(H: ndarray) -> None:
    print(f"H:\n{H}\n")

def print_H_for_all(grid: Grid) -> None:
    for i, element in enumerate(grid.elements):
        print(f"Element {i + 1}:")
        print_H_matrix(element.H)