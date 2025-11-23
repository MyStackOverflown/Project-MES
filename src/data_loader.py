from typing import Tuple, Optional

from models import GlobalData, Grid, Element, Node

def parse_mesh_file(filepath: str) -> Tuple[Optional[GlobalData], Optional[Grid]]:
    data = GlobalData()
    grid = Grid()

    current_section = None

    try:
        with open(filepath, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()

            if not line:
                continue

            if line.startswith("*Node"):
                current_section = "NODE"
                continue
            if line.startswith("*Element"):
                current_section = "ELEMENT"
                continue
            if line.startswith("*BC"):
                current_section = "BC"
                continue

            if current_section is None:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0]
                    val = parts[1]

                    if key == "SimulationTime":
                        data.simulation_time = float(val)
                    elif key == "SimulationStepTime":
                        data.simulation_step_time = float(val)
                    elif key == "Conductivity":
                        data.conductivity = float(val)
                    elif key == "Alfa":
                        data.alfa = float(val)
                    elif key == "Tot":
                        data.tot = float(val)
                    elif key == "InitialTemp":
                        data.initial_temp = float(val)
                    elif key == "Density":
                        data.density = float(val)
                    elif key == "SpecificHeat":
                        data.specific_heat = float(val)
                    elif key == "Nodes" and parts[1] == "number":
                        data.number = int(parts[2])
                    elif key == "Elements" and parts[1] == "number":
                        data.number = int(parts[2])

            elif current_section == "NODE":
                parts = line.split(",")

                if len(parts) < 3:
                    continue

                x = float(parts[1])
                y = float(parts[2])
                grid.nodes.append(Node(x = x, y = y))

            elif current_section == "ELEMENT":
                parts = line.split(",")

                if len(parts) < 5:
                    continue

                node_ids = [int(n.strip()) for n in parts[1:]]
                grid.elements.append(Element(node_ids = node_ids))

        grid.n_nodes = len(grid.nodes)
        grid.n_elements = len(grid.elements)

        data.n_nodes = grid.n_nodes
        data.n_elements = grid.n_elements

        return data, grid

    except FileNotFoundError:
        print("File not found")
        return None, None