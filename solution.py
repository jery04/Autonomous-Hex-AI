from player import Player
from board import HexBoard
from typing import List, Tuple, Optional
import sys


class SmartPlayer(Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph: Optional[HexNodeGraph] = None

    def update_graphs(self, board: HexBoard) -> None:
        """
        Crea lel grarfo si no existe y detecta la última jugada del oponente
        """
        # Reuse existing graphs if present; create them if not.
        if self.graph is None:
            self.graph = HexNodeGraph(size=board.size, player_id=self.player_id)
        
        # Delegate update/opp-detection to the graph itself
        self.graph.update(board)

    def play(self, board: HexBoard) -> tuple:

        best_move = None
        if board.size <= 7:
            # Sincronizar y actualizar grafos a partir del tablero
            self.update_graphs(board) 
            
            # Usar minimax solo en tableros pequenos para controlar coste.
            _, best_move = Minimax.minimax(
                turno=0,
                profundidad=5,
                graph=self.graph,
            )
            #print("")
            
        if best_move is not None:
            self.graph.mark_node_at(*best_move, self.player_id)
            return best_move
        
        # Algo raro
        return (-1, -1)

class Node:
    def __init__(self, r: int, c: int, marked: int = 0):
        self.r = r
        self.c = c
        self.neighbors: List["Node"] = []  # lista de Nodos adyacentes
        self.marked = marked

    def __repr__(self) -> str:
        return f"Node({self.r},{self.c})"

class HexNodeGraph:
    """
    Clase que crea una matriz de `Node` y añade dos nodos extremos.
    - `create_node_matrix(size, orientation)` crea la matriz y enlaza vecinos.
    - `orientation` 1 = extremos izquierda-derecha, 2 = extremos arriba-abajo.
    Los nodos extremos se exponen en `extreme1` y `extreme2`.
    """

    def __init__(self, size: int, player_id: int) -> None:
        self.size = size
        self.player = player_id
        self.matrix: List[List[Node]] = []
        self.node_left: Optional[Node] = None
        self.node_right: Optional[Node] = None
        self.node_up: Optional[Node] = None
        self.node_bottom: Optional[Node] = None
        self.opp: Optional[int] = None
        self.hex_board: Optional[HexBoard] = None
        self.create_node_matrix()

    def detect_opponent_move(self, board: HexBoard) -> Optional[Tuple[int, int]]:
        """
        Recibe un `HexBoard`, guarda una copia en `hex_board` y devuelve
        la coordenada (row, col) de una nueva ficha puesta por el adversario
        (id opuesto a `self.player`) comparando con la última copia guardada.

        - Si no hay `self.player` definido, devuelve None.
        - Si no había tablero previo guardado, sólo guarda el tablero y devuelve None.
        - Si detecta múltiples cambios devuelve la primera encontrada.
        """
        
        prev = self.hex_board.board if self.hex_board else None
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
                if prev[r][c] != curr[r][c] and curr[r][c] == self.opp:
                    self.hex_board = board.clone()
                    return (r, c)

        # si no se detectó nada nuevo, actualizar la copia y devolver None
        self.hex_board = board.clone()
        return None

    def update(self, board: HexBoard) -> None:
        """
        Sincroniza la copia interna del tablero y marca en la gráfica
        la última jugada detectada del oponente (si existe).

        - Llama a `detect_opponent_move` para actualizar `self.hex_board`.
        - Si `detect_opponent_move` devuelve una coordenada, marca el
            nodo correspondiente con `self.opp`.
        """
        opp_move = self.detect_opponent_move(board)
        if opp_move is not None:
            self.mark_node_at(*opp_move, self.opp)

    def is_cell_available(self, r: int, c: int) -> bool:
        """Comprueba la casilla (r,c) en `HexNodeGraph.hex_board`.

        Devuelve True si existe `hex_board` y la posición contiene 0,
        en cualquier otro caso devuelve False (fuera de rango o no inicializado).
        """
        board = self.hex_board
        if board is None:
            return False

        size = board.size

        if not (0 <= r < size and 0 <= c < size):
            return False

        return self.matrix[r][c].marked == 0

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

    def create_node_matrix(self) -> List[List[Node]]:
        """
        Crea la matriz NxN y conecta los nodos extremos.
        `orientation` debe ser 1 (izquierda-derecha) o 2 (arriba-abajo).
        """
        if self.player not in (1, 2):
            raise ValueError("orientation must be 1 (L-R) or 2 (T-B)")

        # Si no hay tablero previo, inicializarlo (variable de clase)
        if self.hex_board is None:
            self.hex_board = HexBoard(size= self.size)

        matrix: List[List[Node]] = [[Node(r, c) for c in range(self.size)] for r in range(self.size)]
        # Guardar el id del jugador, pero no usarlo para decidir qué extremos
        # conectar: creamos los cuatro nodos extremos y los enlazamos todos.
        self.opp = 2 if self.player == 1 else 1

        for r in range(self.size):
            for c in range(self.size):
                neigh_coords = self._neighbors(r, c)
                node = matrix[r][c]
                node.neighbors = []
                for (nr, nc) in neigh_coords:
                    if 0 <= nr < self.size and 0 <= nc < self.size:
                        node.neighbors.append(matrix[nr][nc])

        # Crear nodos extremos (no pertenecen a la matriz)
        self.node_left = Node(-1, -1, 1)
        self.node_right = Node(-2, -2, 1)
        self.node_up = Node(-3, -3, 2)
        self.node_bottom = Node(-4, -4, 2)

        # Conectar extremos izquierda/derecha a las columnas
        for r in range(self.size):
            left = matrix[r][0]
            right = matrix[r][self.size - 1]
            left.neighbors.append(self.node_left)
            self.node_left.neighbors.append(left)
            right.neighbors.append(self.node_right)
            self.node_right.neighbors.append(right)

        # Conectar extremos arriba/abajo a las filas
        for c in range(self.size):
            top = matrix[0][c]
            bottom = matrix[self.size - 1][c]
            top.neighbors.append(self.node_up)
            self.node_up.neighbors.append(top)
            bottom.neighbors.append(self.node_bottom)
            self.node_bottom.neighbors.append(bottom)

        self.matrix = matrix
        return matrix

    def mark_node_at(self, r: int, c: int, player_id: Optional[int] = None) -> None:
        """
        Marca o desmarca el nodo en (r, c).

        - Si `player_id` es 1 o 2, guarda `player_id` en `marked`.
        - Si `player_id` es `None`, guarda 0 en `marked` (desmarca).
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

        if player_id is None:
            node.marked = 0
            return

        if player_id not in (1, 2):
            raise ValueError("player_id must be 1 or 2")

        node.marked = player_id

    def distance_between_extremes(self, player_id: int) -> Optional[int]:
        """
        Retorna la distancia ponderada entre extremos usando 0-1 BFS.

        Si `player_id == 1` calcula entre `node_left` y `node_right`.
        Si `player_id == 2` calcula entre `node_up` y `node_bottom`.

        Reglas de transición:
        - Solo se puede transitar por nodos con `marked` en {0, player_id}.
        - Pasar por un nodo con `marked == player_id` cuesta 0.
        - Pasar por un nodo con `marked == 0` cuesta 1.

        Devuelve None si no existe camino o si los extremos no están definidos.
        """
        if player_id == 1:
            a = self.node_left
            b = self.node_right
        elif player_id == 2:
            a = self.node_up
            b = self.node_bottom
        else:
            return None

        if a is None or b is None:
            return None

        from collections import deque

        q = deque([a])
        distances = {a: 0}

        while q:
            node = q.popleft()
            current_dist = distances[node]

            for neigh in node.neighbors:
                if neigh is None:
                    continue

                neigh_mark = getattr(neigh, "marked", 0)
                if neigh_mark not in (0, player_id):
                    continue

                step = 0 if neigh_mark == player_id else 1
                candidate = current_dist + step

                previous = distances.get(neigh)
                if previous is None or candidate < previous:
                    distances[neigh] = candidate
                    if step == 0:
                        q.appendleft(neigh)
                    else:
                        q.append(neigh)

        if b not in distances:
            return None
        
        #for node in distances:
        #    print(f"{node} {distances[node]}")
            
        result = max(distances[b], 0)
        return result

class Minimax:
    """Contiene la heurística y el algoritmo minimax como métodos estáticos."""

    @staticmethod
    def calculate_heuristic(graph: HexNodeGraph) -> Optional[int]:
        if graph is None:
            return None

        if graph.player not in (1, 2) or graph.opp not in (1, 2):
            return None

        d_self = graph.distance_between_extremes(graph.player)
        d_opponent = graph.distance_between_extremes(graph.opp)
        
        if d_self is None:
            return -1000
        
        if d_opponent is None:
            return 1000

        return d_opponent - d_self

    @staticmethod
    def minimax(
        turno: int,
        profundidad: int,
        graph: HexNodeGraph,
        alpha: int = -sys.maxsize - 1,
        beta: int = sys.maxsize,
        maximizing: bool = True,
        tuple1 = None,
        tuple2 = None,
    ) -> Tuple[int, Optional[Tuple[int, int]]]:
        """
        Versión estática del minimax. Retorna (valor, mejor_jugada).
        """

        if turno >= profundidad-1:
            val = Minimax.calculate_heuristic(graph)
            #print(f"{tuple1} {tuple2} {val}")
            return val, None

        moves = []
        for r in range(graph.size):
            for c in range(graph.size):
                # Una casilla es legal en este estado minimax solo si
                if graph.is_cell_available(r, c):
                    moves.append((r, c))
        
        if not moves:
            return Minimax.calculate_heuristic(graph), None

        best_move = None

        if maximizing:
            max_eval = -sys.maxsize - 1
            for r, c in moves:
                # Marcar la jugada
                graph.mark_node_at(r, c, graph.player)

                eval, _ = Minimax.minimax(turno + 1, profundidad, graph, alpha, beta, False, (r,c))
                
                # Desmarcar después de evaluar
                graph.mark_node_at(r, c, None)  
                
                #print(f"{eval} ({r},{c})")
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

                # Marcar la jugada
                graph.mark_node_at(r, c, graph.opp)

                eval, _ = Minimax.minimax(turno + 1, profundidad, graph, alpha, beta, True, tuple1, (r,c))

                # Desmarcar después de evaluar
                graph.mark_node_at(r, c)

                #print(f"{eval} ({r},{c})")               
                
                if eval is not None and eval < min_eval:
                    min_eval = eval
                beta = min(beta, eval if eval is not None else beta)
                if beta <= alpha:
                    break
            #print("")

            return min_eval, None
