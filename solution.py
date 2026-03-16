from player import Player
from board import HexBoard
from typing import List, Tuple, Optional
import sys


class SmartPlayer(Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.own_graph = None
        self.opp_graph = None

    def update_graphs(self, board: HexBoard) -> None:
        """Ensure graphs exist and sync them with the provided `board`.

        Crea los grafos si no existen, inicializa la copia del tablero en
        `HexNodeGraph.hex_board` y detecta la última jugada del oponente
        actualizando ambos grafos según corresponda.
        """
        # Reuse existing graphs if present; create them if not.
        if self.own_graph is None or self.opp_graph is None:
            self.own_graph = HexNodeGraph()
            self.own_graph.create_node_matrix(board.size, orientation=self.player_id)
            
            opponent_id = 2 if self.player_id == 1 else 1
            self.opp_graph = HexNodeGraph()
            self.opp_graph.create_node_matrix(board.size, orientation=opponent_id)
            
        opp_move = HexNodeGraph.detect_opponent_move(board, self.own_graph.player)
        if opp_move is not None:
            self.opp_graph.mark_node_at(*opp_move)
            self.own_graph.remove_node_at(*opp_move)

    def play(self, board: HexBoard) -> tuple:

        best_move = None
        if board.size <= 7:
            # Sincronizar y actualizar grafos a partir del tablero
            self.update_graphs(board) 
            
            # Usar minimax solo en tableros pequenos para controlar coste.
            _, best_move = Minimax.minimax(
                turno=0,
                size=board.size,
                profundidad=3,
                grafo_propio=self.own_graph,
                grafo_oponente=self.opp_graph,
                alpha=-(sys.maxsize + 1),
                beta=sys.maxsize,
            )

        if best_move is not None:
            return best_move
        
        # Algo raro
        return (-1, -1)

class Node:
    def __init__(self, r: int, c: int, marked: bool = False):
        self.r = r
        self.c = c
        self.neighbors: List["Node"] = []  # lista de Nodos adyacentes
        self.marked = marked

    def __repr__(self) -> str:
        return f"Node({self.r},{self.c})"

    def clone(self) -> "Node":
        """Devuelve una copia superficial del nodo (sin vecinos).

        La lista `neighbors` queda vacía: las referencias se reconstruyen
        posteriormente al clonar el grafo completo.
        """
        n = Node(self.r, self.c)
        n.marked = getattr(self, "marked", False)
        n.neighbors = []
        return n

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
        self.min_distance: Optional[int] = None

    # Campo estático compartido por todas las instancias
    hex_board: Optional[HexBoard] = None

    @staticmethod
    def detect_opponent_move(board: HexBoard, player: Optional[int]) -> Optional[Tuple[int, int]]:
        """
        Recibe un `HexBoard`, guarda una copia en `hex_board` y devuelve
        la coordenada (row, col) de una nueva ficha puesta por el adversario
        (id opuesto a `self.player`) comparando con la última copia guardada.

        - Si no hay `self.player` definido, devuelve None.
        - Si no había tablero previo guardado, sólo guarda el tablero y devuelve None.
        - Si detecta múltiples cambios devuelve la primera encontrada.
        """
        
        if player not in (1, 2):
            return None

        opponent_id = 2 if player == 1 else 1

        prev = HexNodeGraph.hex_board.board
        curr = board.board

        if len(prev) != len(curr):
            # tamaños distintos: actualizar y salir
            HexNodeGraph.hex_board = board.clone()
            return None

        size = len(curr)
        for r in range(size):
            for c in range(size):
                # detectar cualquier cambio en la posición que ahora pertenece
                # al adversario; terminar y devolver la primera encontrada
                if prev[r][c] != curr[r][c] and curr[r][c] == opponent_id:
                    HexNodeGraph.hex_board = board.clone()
                    return (r, c)

        # si no se detectó nada nuevo, actualizar la copia y devolver None
        HexNodeGraph.hex_board = board.clone()
        return None

    @staticmethod
    def is_cell_available(r: int, c: int) -> bool:
        """Comprueba la casilla (r,c) en `HexNodeGraph.hex_board`.

        Devuelve True si existe `hex_board` y la posición contiene 0,
        en cualquier otro caso devuelve False (fuera de rango o no inicializado).
        """
        board = HexNodeGraph.hex_board
        if board is None:
            return False

        size = board.size

        if not (0 <= r < size and 0 <= c < size):
            return False

        return board.board[r][c] == 0

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

        # Si no hay tablero previo, almacenar y salir (variable de clase)
        if HexNodeGraph.hex_board is None:
            HexNodeGraph.hex_board = HexBoard(size=size)
            return None

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
        self.extreme1 = Node(-1, -1, True)
        self.extreme2 = Node(-2, -2, True)

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

    def mark_node_at(self, r: int, c: int) -> None:
        """
        Marca el nodo en (r, c) estableciendo su atributo `marked = True`.

        - Lanza `IndexError` si la matriz no existe o las coordenadas están
          fuera de rango.
        - Lanza `ValueError` si la posición ya es `None` (nodo eliminado).
        """
        if not self.matrix:
            raise IndexError("matrix is empty")
        size = len(self.matrix)
        if not (0 <= r < size and 0 <= c < size):
            raise IndexError("coordinates out of range")

        node = self.matrix[r][c]
        if node is None:
            raise ValueError("node at given coordinates is None")

        node.marked = True

    def clone_and_mark(self, r: int, c: int) -> "HexNodeGraph":
        """
        Clona este `HexNodeGraph` usando `clone()` y marca la posición (r, c)
        en la copia. Devuelve la copia modificada.

        - El método realiza la clonación primero y luego usa
          `mark_node_at` sobre la nueva instancia para marcar.
        - Propaga las mismas excepciones que `mark_node_at` si las
          coordenadas no son válidas o la posición es `None`.
        """
        new_graph = self.clone()
        new_graph.mark_node_at(r, c)
        return new_graph

    def clone_and_remove(self, r: int, c: int) -> "HexNodeGraph":
        """
        Clona este `HexNodeGraph` y elimina la posición (r, c) en la copia.

        - Realiza `clone()` y después llama a `remove_node_at(r, c)` sobre la
            copia para no modificar el grafo original.
        - Propaga las mismas excepciones que `remove_node_at` si las
            coordenadas no son válidas o la posición ya es `None`.
        """
        new_graph = self.clone()
        new_graph.remove_node_at(r, c)
        return new_graph

    def distance_between_extremes(self) -> Optional[int]:
        """
        Retorna la distancia ponderada entre `extreme1` y `extreme2` usando
        0-1 BFS:

        - Moverse hacia un nodo con `marked == True` cuesta 0.
        - Moverse hacia un nodo con `marked == False` cuesta 1.

        - If `stop_on_first` is True (default) the search returns immediately
          when `extreme2` is first encountered.
        - If `stop_on_first` is False the search computes the global minimum.

        Devuelve None si no existe camino o si los extremos no están definidos.
        """
        if self.extreme1 is None or self.extreme2 is None:
            return None

        from collections import deque

        q = deque([self.extreme1])
        distances = {self.extreme1: 0}

        while q:
            node = q.popleft()
            current_dist = distances[node]

            for neigh in node.neighbors:
                if neigh is None:
                    continue

                step = 0 if getattr(neigh, "marked", False) else 1
                candidate = current_dist + step

                previous = distances.get(neigh)
                if previous is None or candidate < previous:
                    distances[neigh] = candidate
                    if step == 0:
                        q.appendleft(neigh)
                    else:
                        q.append(neigh)

        if self.extreme2 not in distances:
            self.min_distance = None
            return None

        result = max(distances[self.extreme2], 0)
        self.min_distance = result
        return result

    def clone(self) -> "HexNodeGraph":
        """
        Devuelve una copia profunda de este `HexNodeGraph`.

        - Se copian todos los `Node` existentes en `matrix` (las posiciones
          que sean `None` se mantienen como `None`).
        - Se copian las referencias de vecinos apuntando a los nuevos nodos
          correspondientes (no se comparten objetos con el original).
        - Se clonan `extreme1` y `extreme2` si existen y se mantienen las
          conexiones adecuadas.
        - Se copia `player`, `min_distance` y se clona `hex_board` si está
          presente (usando su método `clone`).
        """
        new = HexNodeGraph()

        new.player = self.player
        new.min_distance = self.min_distance

        if self.hex_board is not None:
            try:
                # `hex_board` ahora es campo de clase; no clonarlo por instancia
                pass
            except Exception:
                pass

        if not self.matrix:
            return new

        size = len(self.matrix)
        new_matrix: List[List[Optional[Node]]] = [[None for _ in range(size)] for _ in range(size)]
        mapping: dict = {}

        # crear nodos clonados para cada posición (o None)
        for r in range(size):
            for c in range(size):
                old = self.matrix[r][c]
                if old is None:
                    new_matrix[r][c] = None
                else:
                    cloned = old.clone()
                    new_matrix[r][c] = cloned
                    mapping[old] = cloned

        # clonar extremos si existen
        if self.extreme1 is not None:
            mapping[self.extreme1] = self.extreme1.clone()
            new.extreme1 = mapping[self.extreme1]
        if self.extreme2 is not None:
            mapping[self.extreme2] = self.extreme2.clone()
            new.extreme2 = mapping[self.extreme2]

        # reconstruir referencias de vecinos usando el mapping
        for old_node, new_node in mapping.items():
            for old_neigh in getattr(old_node, "neighbors", []):
                if old_neigh is None:
                    continue
                target = mapping.get(old_neigh)
                if target is not None:
                    new_node.neighbors.append(target)

        new.matrix = new_matrix
        return new

class Minimax:
    """Contiene la heurística y el algoritmo minimax como métodos estáticos."""

    @staticmethod
    def calculate_heuristic(self_graph: HexNodeGraph, opponent_graph: HexNodeGraph) -> Optional[int]:
        if self_graph is None or opponent_graph is None:
            return None

        d_self = self_graph.distance_between_extremes()
        d_opponent = opponent_graph.distance_between_extremes()

        if d_self is None or d_opponent is None:
            return None

        return d_opponent - d_self

    @staticmethod
    def minimax(
        turno: int,
        size: int,
        profundidad: int,
        grafo_propio: HexNodeGraph,
        grafo_oponente: HexNodeGraph,
        alpha: int = -sys.maxsize - 1,
        beta: int = sys.maxsize,
        maximizing: bool = True,
    ) -> Tuple[int, Optional[Tuple[int, int]]]:
        """
        Versión estática del minimax. Retorna (valor, mejor_jugada).
        """

        if turno >= profundidad:
            val = Minimax.calculate_heuristic(grafo_propio, grafo_oponente)
            return val, None

        moves = []
        for r in range(size):
            for c in range(size):
                if HexNodeGraph.is_cell_available(r, c):
                    moves.append((r, c))

        if not moves:
            return Minimax.calculate_heuristic(grafo_propio, grafo_oponente), None

        best_move = None

        if maximizing:
            max_eval = -sys.maxsize - 1
            for r, c in moves:
                clonado_p = grafo_propio.clone_and_mark(r, c)
                clonado_o = grafo_oponente.clone_and_remove(r, c)

                eval, _ = Minimax.minimax(turno + 1, size, profundidad, clonado_p, clonado_o, alpha, beta, False)
                if eval is not None and eval > max_eval:
                    max_eval = eval
                    if turno == 0:
                        best_move = (r, c)
                alpha = max(alpha, eval if eval is not None else alpha)
                if beta <= alpha:
                    break
            return max_eval, best_move

        else:
            min_eval = sys.maxsize
            for r, c in moves:
                clonado_o = grafo_oponente.clone_and_mark(r, c)
                clonado_p = grafo_propio.clone_and_remove(r, c)

                eval, _ = Minimax.minimax(turno + 1, size, profundidad, clonado_p, clonado_o, alpha, beta, True)
                if eval is not None and eval < min_eval:
                    min_eval = eval
                beta = min(beta, eval if eval is not None else beta)
                if beta <= alpha:
                    break
            return min_eval, None
