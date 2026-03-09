from dataclasses import dataclass, field
from typing import List, Optional

from numpy import ndarray

@dataclass
class GlobalData:
    simulation_time: float = 0.0
    simulation_step_time: float = 0.0
    conductivity: float = 0.0
    alfa: float = 0.0
    tot: float = 0.0
    initial_temp: float = 0.0
    density: float = 0.0
    specific_heat: float = 0.0
    n_nodes: int = 0
    n_elements: int = 0

    def summary(self) -> str:
        return (
            f"Global Data:\n"
            f"SimulationTime: {self.simulation_time}\n"
            f"SimulationStepTime: {self.simulation_step_time}\n"
            f"Conductivity: {self.conductivity}\n"
            f"Alfa: {self.alfa}\n"
            f"Tot: {self.tot}\n"
            f"InitialTemp: {self.initial_temp}\n"
            f"Density: {self.density}\n"
            f"SpecificHeat: {self.specific_heat}\n"
            f"Nodes number: {self.n_nodes}\n"
            f"Elements number: {self.n_elements}\n"
        )

@dataclass
class Grid:
    n_nodes: int = 0
    n_elements: int = 0
    nodes: List["Node"] = field(default_factory=list)
    elements: List["Element"] = field(default_factory=list)

    def summary(self) -> str:
        s = f"Grid summary: {self.n_nodes} nodes, {self.n_elements} elements\n"
        s += "\nNodes:\n"
        for node in self.nodes:
            s += f"\n{node.summary()}"
        s += "\nElements:\n"
        for element in self.elements:
            s += f"\n{element.summary()}"
        return s

@dataclass
class Element:
    node_ids: List[int]
    jacobians: List["Jacobian"] = field(default_factory=list)
    H: Optional[ndarray] = None
    Hbc: Optional[ndarray] = None
    P: Optional[ndarray] = None
    C: Optional[ndarray] = None

    def summary(self) -> str:
        return f"Nodes = {self.node_ids}"

@dataclass
class Node:
    x: float = 0.0
    y: float = 0.0
    bc: bool = False # czy war brzeg?

    def summary(self) -> str:
        return f"bc = {self.bc}\n(x = {self.x}\ny = {self.y})\n"