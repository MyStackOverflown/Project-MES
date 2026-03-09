from typing import List
from dataclasses import dataclass

from numpy import ndarray, array, linalg, fromiter

from src.datamodels.models import Grid
from src.jacobian.elem_univ import ElemUniv

@dataclass
class Jacobian:
    J: ndarray
    J_inv: ndarray
    J_det: float

def jacobian_for_element(x: ndarray, y: ndarray, elem_univ: ElemUniv, n: int = 2) -> List[Jacobian]:
    jacobians = []

    for i in range(n * n):
        dN_dXi = elem_univ.dN_dXi[i]
        dN_dEta = elem_univ.dN_dEta[i]

        J11 = dN_dXi @ x
        J12 = dN_dXi @ y
        J21 = dN_dEta @ x
        J22 = dN_dEta @ y

        J = array([[J11, J12], [J21, J22]])
        J_det = linalg.det(J)
        J_inv = linalg.inv(J)

        jacobians.append(Jacobian(J, J_inv, J_det))

    return jacobians

def jacobian_for_all(grid: Grid, elem_univ: ElemUniv, n: int = 2) -> None:
    all_x = fromiter((n.x for n in grid.nodes), dtype=float)
    all_y = fromiter((n.y for n in grid.nodes), dtype=float)

    for element in grid.elements:
        id = array(element.node_ids) - 1
        x = all_x[id]
        y = all_y[id]

        element.jacobians = jacobian_for_element(x, y, elem_univ, n)

def print_jacobian(j: Jacobian) -> None:
    print(f"J_det:\n{j.J_det}")
    print(f"J:\n{j.J}")
    print(f"J_inv:\n{j.J_inv}")

def print_jacobian_for_all(grid: Grid) -> None:
    for i, element in enumerate(grid.elements):
        print(f"\nElement {i + 1}:\n")
        for k, j in enumerate(element.jacobians):
            print(f"Point {k + 1}:")
            print_jacobian(j)