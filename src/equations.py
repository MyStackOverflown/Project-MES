from dataclasses import dataclass

from numpy import ndarray

@dataclass
class GlobalMatrix:
    H_global: ndarray