from dataclasses import dataclass

from numpy import ndarray, polynomial, zeros

@dataclass
class ElemUniv:
    dN_dXi: ndarray
    dN_dEta: ndarray
    N: ndarray
    W: ndarray # wagi niezgodnie z wytycznymi ale tak jest lepiej (lukasz tak ma) UPDATE: mozna wywalic ale moze cos nie dzialac wiec na razie zostaje

def build_elem_univ(n: int = 2) -> ElemUniv:
    pc, wc = polynomial.legendre.leggauss(n)

    dN_dXi = zeros((n * n, 4))
    dN_dEta = zeros((n * n, 4))
    N = zeros((n * n, 4))
    W = zeros(n * n)

    id = 0
    for i in range(n):
        for j in range(n):
            xi = pc[i]
            eta = pc[j]
            W[id] = wc[i] * wc[j]

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

            N[id, 0] = 0.25 * (1 - xi) * (1 - eta)
            N[id, 1] = 0.25 * (1 + xi) * (1 - eta)
            N[id, 2] = 0.25 * (1 + xi) * (1 + eta)
            N[id, 3] = 0.25 * (1 - xi) * (1 + eta)

            id += 1

    return ElemUniv(dN_dXi, dN_dEta, N, W)