import sys
import random
from solution import HexNodeGraph


def check_graph(size: int, orientation: int):
    g = HexNodeGraph()
    matrix = g.create_node_matrix(size, orientation)

    # tamaño
    if len(matrix) != size or any(len(row) != size for row in matrix):
        raise AssertionError("Matriz con tamaño incorrecto")

    # consistencia bidireccional de vecinos
    for r in range(size):
        for c in range(size):
            node = matrix[r][c]
            for nb in node.neighbors:
                if node not in nb.neighbors:
                    raise AssertionError(f"Vecino no bidireccional: {node} -> {nb}")

    # extremos bien conectados según orientación
    if orientation == 1:
        for r in range(size):
            left = matrix[r][0]
            right = matrix[r][size - 1]
            if g.extreme1 not in left.neighbors or left not in g.extreme1.neighbors:
                raise AssertionError(f"Extremo izquierdo mal conectado en fila {r}")
            if g.extreme2 not in right.neighbors or right not in g.extreme2.neighbors:
                raise AssertionError(f"Extremo derecho mal conectado en fila {r}")
    else:
        for c in range(size):
            top = matrix[0][c]
            bottom = matrix[size - 1][c]
            if g.extreme1 not in top.neighbors or top not in g.extreme1.neighbors:
                raise AssertionError(f"Extremo superior mal conectado en columna {c}")
            if g.extreme2 not in bottom.neighbors or bottom not in g.extreme2.neighbors:
                raise AssertionError(f"Extremo inferior mal conectado en columna {c}")


def print_node_and_neighbors(node):
    # Maneja None u objetos sin neighbors
    if node is None:
        print("None")
        return
    neighs = getattr(node, "neighbors", None)
    if neighs is None:
        print(f"{node}: no tiene atributo 'neighbors'")
        return
    print(f"{node}: vecinos -> {[n for n in neighs]}")


def print_random_nodes_and_extremes(size: int = 5, orientation: int = 1, count: int = 5):
    g = HexNodeGraph()
    matrix = g.create_node_matrix(size, orientation)

    all_nodes = [n for row in matrix for n in row]
    if not all_nodes:
        print("Tablero vacío")
        return

    print(f"Imprimiendo {count} nodos aleatorios (tablero {size}x{size}, orientación={orientation}):")
    picks = random.sample(all_nodes, min(count, len(all_nodes)))
    for n in picks:
        print_node_and_neighbors(n)

    # Imprimir vecinos de los nodos extremos
    print("\nVecinos de extreme1:")
    print_node_and_neighbors(g.extreme1)
    print("\nVecinos de extreme2:")
    print_node_and_neighbors(g.extreme2)


if __name__ == "__main__":
    try:
        # pruebas básicas
        check_graph(3, 1)
        check_graph(4, 2)

        # orientación inválida -> ValueError
        ok = False
        try:
            HexNodeGraph().create_node_matrix(3, orientation=99)
        except ValueError:
            ok = True
        if not ok:
            raise AssertionError("No se lanzó ValueError para orientación inválida")

        print("OK: HexNodeGraph pasa las comprobaciones básicas\n")

        # Imprimir nodos aleatorios y vecinos de extremos
        print_random_nodes_and_extremes(size=5, orientation=2, count=6)

        sys.exit(0)
    except Exception as e:
        print("ERROR en comprobación:", e)
        sys.exit(2)
