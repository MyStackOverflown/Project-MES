from dataclasses import dataclass

from numpy import ndarray, polynomial, zeros

@dataclass
class ElemUniv:
    dN_dXi: ndarray
    dN_dEta: ndarray
    W: ndarray# wagi niezgodnie z wytycznymi ale tak jest lepiej (lukasz tak ma)

def build_elem_univ(n: int = 2) -> ElemUniv:
    pc, wc = polynomial.legendre.leggauss(n)
    num_integration_points = n * n # dla 2D kwadrat

    dN_dXi = zeros((num_integration_points, 4))
    dN_dEta = zeros((num_integration_points, 4))
    W = zeros(num_integration_points)

    id = 0
    for i in range(n):
        for j in range(n):
            xi = pc[j]
            eta = pc[i]
            W[id] = wc[j] * wc[i]

            # znaki dla ksztaltu
            #--
            #+-
            #++
            #-+

            dN_dXi[id, 0] = -0.25 * (1 - eta)
            dN_dXi[id, 1] = 0.25 * (1 - eta)
            dN_dXi[id, 2] = 0.25 * (1 + eta)
            dN_dXi[id, 3] = -0.25 * (1 + eta)

            dN_dEta[id, 0] = -0.25 * (1 - xi)
            dN_dEta[id, 1] = -0.25 * (1 + xi)
            dN_dEta[id, 2] = 0.25 * (1 + xi)
            dN_dEta[id, 3] = 0.25 * (1 - xi)

            id += 1

    return ElemUniv(dN_dXi, dN_dEta, W)