from player import Player
from board import HexBoard
from typing import List, Tuple, Optional


class SmartPlayer(Player):
    def play(self, board: HexBoard) -> tuple:
        #  Tu lógica aquí
        pass


class Node:
    def __init__(self, r: int, c: int):
        self.r = r
        self.c = c
        self.neighbors: List["Node"] = []  # lista de Nodos adyacentes

    def __repr__(self) -> str:
        return f"Node({self.r},{self.c})"


class HexNodeGraph:
    """
    Clase que crea una matriz de `Node` y añade dos nodos extremos.
    - `create_node_matrix(size, orientation)` crea la matriz y enlaza vecinos.
    - `orientation` 1 = extremos izquierda-derecha, 2 = extremos arriba-abajo.
    Los nodos extremos se exponen en `extreme1` y `extreme2`.
    """

    def __init__(self) -> None:
        self.matrix: List[List[Node]] = []
        self.extreme1: Optional[Node] = None
        self.extreme2: Optional[Node] = None
        self.player: Optional[int] = None

    def _neighbors(self, r: int, c: int) -> List[Tuple[int, int]]:
        # Vecinos en un tablero hexagonal (offset coordinates).
        if r % 2 != 0:  # fila par
            return [
                (r - 1, c - 1),    # arriba-izquierda   NW
                (r - 1, c    ),    # arriba-derecha     NE
                (r    , c + 1),    # derecha            E
                (r + 1, c    ),    # abajo-derecha      SE
                (r + 1, c - 1),    # abajo-izquierda    SW
                (r    , c - 1),    # izquierda          W
            ]
        else:  # fila impar
            return [
                (r - 1, c    ),    # arriba-izquierda   NW
                (r - 1, c + 1),    # arriba-derecha     NE
                (r    , c + 1),    # derecha            E
                (r + 1, c + 1),    # abajo-derecha      SE
                (r + 1, c    ),    # abajo-izquierda    SW
                (r    , c - 1),    # izquierda          W
            ]

    def create_node_matrix(self, size: int, orientation: int = 1) -> List[List[Node]]:
        """Crea la matriz NxN y conecta los nodos extremos.

        `orientation` debe ser 1 (izquierda-derecha) o 2 (arriba-abajo).
        """
        if orientation not in (1, 2):
            raise ValueError("orientation must be 1 (L-R) or 2 (T-B)")

        matrix: List[List[Node]] = [[Node(r, c) for c in range(size)] for r in range(size)]
        self.player = orientation

        for r in range(size):
            for c in range(size):
                neigh_coords = self._neighbors(r, c)
                node = matrix[r][c]
                node.neighbors = []
                for (nr, nc) in neigh_coords:
                    if 0 <= nr < size and 0 <= nc < size:
                        node.neighbors.append(matrix[nr][nc])

        # Crear nodos extremos (no pertenecen a la matriz)
        self.extreme1 = Node(-1, -1)
        self.extreme2 = Node(-2, -2)
        self.extreme1.neighbors = []
        self.extreme2.neighbors = []

        if orientation == 1:
            # Conectar extremos a las columnas izquierda y derecha
            for r in range(size):
                left = matrix[r][0]
                right = matrix[r][size - 1]
                left.neighbors.append(self.extreme1)
                self.extreme1.neighbors.append(left)
                right.neighbors.append(self.extreme2)
                self.extreme2.neighbors.append(right)
        else:
            # Conectar extremos a las filas superior e inferior
            for c in range(size):
                top = matrix[0][c]
                bottom = matrix[size - 1][c]
                top.neighbors.append(self.extreme1)
                self.extreme1.neighbors.append(top)
                bottom.neighbors.append(self.extreme2)
                self.extreme2.neighbors.append(bottom)

        self.matrix = matrix
        return matrix

    def remove_node_at(self, r: int, c: int, verbose: bool = True) -> None:
        """
        Elimina el nodo en (r, c) de la matriz y quita su referencia
        de todos sus adyacentes. Después pone la posición en la matriz a None.
        Si la posición ya es None no hace nada. Lanza IndexError si las
        coordenadas están fuera de rango o la matriz no existe.
        """
        if not self.matrix:
            raise IndexError("matrix is empty")
        size = len(self.matrix)
        if not (0 <= r < size and 0 <= c < size):
            raise IndexError("coordinates out of range")

        node = self.matrix[r][c]
        if node is None:
            return

        # Remover la referencia del nodo en cada vecino
        neighs = list(node.neighbors)
        for neigh in neighs:
            try:
                neigh.neighbors.remove(node)
            except ValueError:
                pass

        # Limpiar las conexiones del nodo y eliminar de la matriz
        node.neighbors = []
        self.matrix[r][c] = None

        if verbose:
            # Mostrar la lista de adyacencia actual de los antiguos vecinos
            print(f"Removed node at ({r},{c}). Neighbors' adjacency:")
            for n in neighs:
                # imprimir la adyacencia tal cual se guarda en el objeto Node
                adj = getattr(n, "neighbors", None)

                # comprobar qué hay en la posición de la matriz (si aplica)
                if 0 <= n.r < size and 0 <= n.c < size:
                    in_matrix = self.matrix[n.r][n.c]
                else:
                    in_matrix = "(not in matrix)"

                print(f" - {n} -> neighbors: {adj}; matrix[{n.r}][{n.c}] = {in_matrix}")


