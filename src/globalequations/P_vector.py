from numpy import ndarray, zeros, polynomial, sqrt, array, fromiter, newaxis

from src.datamodels.models import GlobalData, Grid

def P_vector_for_element(x: ndarray, y: ndarray, bc: ndarray, alfa: float, tot: float, n: int = 2) -> ndarray:
    P_for_element = zeros(4)

    pc, wc = polynomial.legendre.leggauss(n)

    sides = [(0, 1), (1, 2), (2, 3), (3, 0)] # +1 do kazdej wartosci i jest jak w prezce

    for side_id, (id1, id2) in enumerate(sides):
        if bc[id1] == True and bc[id2] == True:
            dx = x[id1] - x[id2]
            dy = y[id1] - y[id2]

            a = sqrt(dx ** 2 + dy ** 2)
            j_det = 0.5 * a

            for i in range(n):
                p = pc[i]
                w = wc[i]

                # \/ > /\ <

                if side_id == 0:
                    xi, eta = p, -1
                elif side_id == 1:
                    xi, eta = 1, p
                elif side_id == 2:
                    xi, eta = -p, 1
                elif side_id == 3:
                    xi, eta = -1, -p

                #--
                #+-
                #++
                #-+

                N = array([
                    0.25 * (1 - xi) * (1 - eta),
                    0.25 * (1 + xi) * (1 - eta),
                    0.25 * (1 + xi) * (1 + eta),
                    0.25 * (1 - xi) * (1 + eta),
                ])

                P_for_element += alfa * N * tot * j_det * w

    return P_for_element

def P_vector_for_all(grid: Grid, global_data: GlobalData, n: int = 2):
    all_x = fromiter([n.x for n in grid.nodes], dtype=float)
    all_y = fromiter([n.y for n in grid.nodes], dtype=float)
    all_bc = fromiter([n.bc for n in grid.nodes], dtype=bool)

    for element in grid.elements:
        id = array(element.node_ids) - 1
        x = all_x[id]
        y = all_y[id]
        bc = all_bc[id]

        element.P = P_vector_for_element(x, y, bc, global_data.alfa, global_data.tot, n)

def print_P_vector(P: ndarray) -> None:
    print(f"P:\n{P[:, newaxis]}\n")

def print_P_for_all(grid: Grid) -> None:
    for i, element in enumerate(grid.elements):
        print(f"Element {i + 1}:")
        print_P_vector(element.P)