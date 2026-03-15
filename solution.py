from player import Player
from board import HexBoard
from typing import List, Tuple, Optional
import sys


class SmartPlayer(Player):
    def play(self, board: HexBoard) -> tuple:
        own_graph = HexNodeGraph()
        own_graph.create_node_matrix(board.size, orientation=self.player_id)

        opponent_id = 2 if self.player_id == 1 else 1
        opp_graph = HexNodeGraph()
        opp_graph.create_node_matrix(board.size, orientation=opponent_id)

        best_move = None
        if board.size <= 7:
            # Usar minimax solo en tableros pequenos para controlar coste.
            _, best_move = minimax(
                turno=0,
                profundidad=3,
                grafo_propio=own_graph,
                grafo_oponente=opp_graph,
                best_alpha=-(sys.maxsize + 1),
                best_beta=sys.maxsize,
                best_casilla=None,
            )

        if best_move is not None:
            return best_move
        
        # Algo raro
        return (-1, -1)

class Node:
    def __init__(self, r: int, c: int):
        self.r = r
        self.c = c
        self.neighbors: List["Node"] = []  # lista de Nodos adyacentes
        self.marked = False

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
        self.hex_board: Optional[HexBoard] = None
        self.min_distance: Optional[int] = None

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
        self.extreme1.marked = True
        self.extreme2.marked = True

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

    def detect_opponent_move(self, board: HexBoard) -> Optional[Tuple[int, int]]:
        """
        Recibe un `HexBoard`, guarda una copia en `self.hex_board` y devuelve
        la coordenada (row, col) de una nueva ficha puesta por el adversario
        (id opuesto a `self.player`) comparando con la última copia guardada.

        - Si no hay `self.player` definido, devuelve None.
        - Si no había tablero previo guardado, sólo guarda el tablero y devuelve None.
        - Si detecta múltiples cambios devuelve la primera encontrada.
        """
        if self.player not in (1, 2):
            return None

        opponent_id = 2 if self.player == 1 else 1

        # Si no hay tablero previo, almacenar y salir
        if self.hex_board is None:
            self.hex_board = board.clone()
            return None

        prev = self.hex_board.board
        curr = board.board

        if len(prev) != len(curr):
            # tamaños distintos: actualizar y salir
            self.hex_board = board.clone()
            return None

        size = len(curr)
        for r in range(size):
            for c in range(size):
                # detectar cualquier cambio en la posición que ahora pertenece
                # al adversario; terminar y devolver la primera encontrada
                if prev[r][c] != curr[r][c] and curr[r][c] == opponent_id:
                    self.hex_board = board.clone()
                    return (r, c)

        # si no se detectó nada nuevo, actualizar la copia y devolver None
        self.hex_board = board.clone()
        return None

    def add_adjacents_to_node(self, r: int, c: int, target: Optional[Node] = None) -> None:
        """
        Añade todos los nodos adyacentes del nodo en (r, c) a `target`.

        - `target` por defecto es `self.extreme1`.
                - Evita duplicados en `target.neighbors`.
                - No modifica la lista de vecinos de los nodos adyacentes (no
                    se añade `target` a `neigh.neighbors`).
        - Lanza `IndexError` si la matriz no existe o coordenadas fuera de
          rango, y `ValueError` si el nodo fuente o el target son `None`.
        """
        if not self.matrix:
            raise IndexError("matrix is empty")
        size = len(self.matrix)
        if not (0 <= r < size and 0 <= c < size):
            raise IndexError("coordinates out of range")

        source = self.matrix[r][c]
        if source is None:
            raise ValueError("source node is None")

        if target is None:
            target = self.extreme1
        if target is None:
            raise ValueError("target node is None")

        # Añadir cada vecino del source al target sin duplicados (no viceversa)
        for neigh in list(source.neighbors):
            if neigh is None or neigh is source or neigh is target:
                continue

            if neigh not in target.neighbors:
                target.neighbors.append(neigh)

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

    def is_any_adjacent_marked(self, r: int, c: int) -> bool:
        """
        Devuelve True si alguno de los nodos adyacentes al nodo en (r, c)
        tiene `marked == True`. Levanta `IndexError` si la matriz no existe o
        las coordenadas están fuera de rango, y `ValueError` si la posición
        (r,c) es `None` (nodo eliminado).
        """
        if not self.matrix:
            raise IndexError("matrix is empty")
        size = len(self.matrix)
        if not (0 <= r < size and 0 <= c < size):
            raise IndexError("coordinates out of range")

        source = self.matrix[r][c]
        if source is None:
            raise ValueError("source node is None")

        for neigh in source.neighbors:
            if neigh is None:
                continue
            if getattr(neigh, "marked", False):
                return True
        return False

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

    def distance_between_extremes(self, stop_on_first: bool = True) -> Optional[int]:
        """
        Retorna la distancia (número de aristas) entre `extreme1` y `extreme2`
        usando BFS.

        - If `stop_on_first` is True (default) the search returns immediately
          when `extreme2` is first encountered.
        - If `stop_on_first` is False the BFS continues and the first-found
          distance is recorded and returned after the search finishes.

        Devuelve None si no existe camino o si los extremos no están definidos.
        """
        if self.extreme1 is None or self.extreme2 is None:
            return None

        from collections import deque

        q = deque()
        q.append((self.extreme1, 0))
        visited = set()
        visited.add((self.extreme1.r, self.extreme1.c))

        found_distance: Optional[int] = None

        while q:
            node, dist = q.popleft()
            if node is self.extreme2:
                current_dist = dist - 1
                if stop_on_first:
                    self.min_distance = current_dist
                    return current_dist
                # record and continue exploring
                if found_distance is None or current_dist < found_distance:
                    found_distance = current_dist

            for neigh in node.neighbors:
                if neigh is None:
                    continue
                coord = (neigh.r, neigh.c)
                if coord in visited:
                    continue
                visited.add(coord)
                q.append((neigh, dist + 1))

        # guardar el resultado encontrado (puede ser None) antes de devolver
        self.min_distance = found_distance
        return found_distance

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
                new.hex_board = self.hex_board.clone()
            except Exception:
                new.hex_board = None

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


def calculate_heuristic(self_graph: HexNodeGraph, opponent_graph: HexNodeGraph) -> Optional[int]:
    """
    Calcula la heurística basada en la distancia entre extremos de dos
    `HexNodeGraph` distintos.

    Parámetros:
    - `self_graph`: el `HexNodeGraph` del propio jugador.
    - `opponent_graph`: el `HexNodeGraph` del adversario.

    Devuelve `distance(opponent) - distance(self)` si ambas distancias
    están definidas; en caso contrario devuelve `None`.
    """
    if self_graph is None or opponent_graph is None:
        return None

    d_self = self_graph.distance_between_extremes()
    d_opponent = opponent_graph.distance_between_extremes()

    if d_self is None or d_opponent is None:
        return None

    return d_opponent - d_self

def _is_cell_available(
    grafo_propio: HexNodeGraph,
    grafo_oponente: HexNodeGraph,
    r: int,
    c: int,
) -> bool:
    node_self = grafo_propio.matrix[r][c]
    node_opp = grafo_oponente.matrix[r][c]
    if node_self is None or node_opp is None:
        return False
    return (not node_self.marked) and (not node_opp.marked)

def minimax(
    turno: int,
    profundidad: int,
    grafo_propio: HexNodeGraph,
    grafo_oponente: HexNodeGraph,
    alpha: int = -sys.maxsize - 1,
    beta: int = sys.maxsize,
    maximizing: bool = True,          # más claro que turno % 2
) -> Tuple[int, Optional[Tuple[int, int]]]:
    """
    Retorna (valor, mejor_jugada) — la jugada solo es válida en la raíz (turno=0)
    """
    size = grafo_propio.size  # asumo que tienes .size

    # 1. Profundidad máxima → hoja
    if turno >= profundidad:
        val = calculate_heuristic(grafo_propio, grafo_oponente)
        return val, None

    # 2. Obtener movimientos legales (lista para poder ordenar)
    moves = []
    for r in range(size):
        for c in range(size):
            if _is_cell_available(grafo_propio, grafo_oponente, r, c):
                moves.append((r, c))

    if not moves:
        # Tablero lleno sin ganador (raro en HEX, pero por si acaso)
        return calculate_heuristic(grafo_propio, grafo_oponente), None

    # 3. Ordenar movimientos (crucial para α-β)
    # Ejemplo simple: ordenar por cercanía a bordes o por heurística rápida
    # moves.sort(key=lambda m: quick_score_move(grafo_propio, grafo_oponente, m[0], m[1]), reverse=maximizing)

    best_move = None

    if maximizing:  # turno propio (MAX)
        max_eval = -sys.maxsize - 1
        for r, c in moves:
            clonado_p = grafo_propio.clone_and_mark(r, c)
            clonado_o = grafo_oponente.clone_and_remove(r, c)
           
            # Si la nueva casilla está adyacente a alguna marcada,
            # propagar los adyacentes al extremo correspondiente.
            if clonado_p.is_any_adjacent_marked(r, c):
                clonado_p.add_adjacents_to_node(r, c)

            eval, _ = minimax(turno + 1, profundidad, clonado_p, clonado_o, alpha, beta, False)
            if eval > max_eval:
                max_eval = eval
                if turno == 0:
                    best_move = (r, c)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # corte α-β
        return max_eval, best_move

    else:  # turno oponente (MIN)
        min_eval = sys.maxsize
        for r, c in moves:
            clonado_o = grafo_oponente.clone_and_mark(r, c)
            clonado_p = grafo_propio.clone_and_remove(r, c)  
            
            # Si la nueva casilla está adyacente a alguna marcada,
            # propagar los adyacentes al extremo correspondiente.
            if clonado_o.is_any_adjacent_marked(r, c):
                clonado_o.add_adjacents_to_node(r, c)
            
            eval, _ = minimax(turno + 1, profundidad, clonado_p, clonado_o, alpha, beta, True)
            if eval < min_eval:
                min_eval = eval
            beta = min(beta, eval)
            if beta <= alpha:
                break  # corte
        return min_eval, None
