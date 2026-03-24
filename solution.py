from player import Player
from board import HexBoard
from typing import List, Tuple, Optional, Set, Iterable
import typing
import sys
import math
import random
from collections import deque
from mcts import MCTS
import time

# LLamada Principal
#-----------------------------------------------------------------------
class SmartPlayer(Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph: Optional[HexGraph] = None
        self.mcts: Optional[MCTS] = None

    def play(self, board: HexBoard) -> tuple:

        if board.size <= 11:
            # Resetear MCTS
            self.mcts = None
            
            # Sincronizar y actualizar grafos a partir del tablero
            if self.graph is None or self.graph.is_different_board(board):
                self.graph = HexGraph(size=board.size, player_id=self.player_id)
                
            # Usar preminimax para elegir profundidad según tamaño
            return Minimax.preminimax(self.graph, board)
        else:
            # Resetear HexGraph
            self.graph = None 
            
            # Sincronizar y actualizar MCTS a partir del tablero
            if self.mcts is None or self.mcts.is_different_board(board):
                self.mcts = MCTS(size=board.size, player_id=self.player_id)
            
            return self.mcts.best_move(board)

        
        # Algo raro
        return (-1, -1)

# Primary Strategy
#-----------------------------------------------------------------------
class Node:
    def __init__(self, r: int, c: int, marked: int = 0):
        self.r = r
        self.c = c
        self.neighbors: List["Node"] = []  # lista de Nodos adyacentes
        self.marked = marked
        self.id_comp: Optional[int] = None  # id de componente conexa (player, comp_id)

    def __repr__(self) -> str:
        return f"Node({self.r},{self.c})"

class HexGraph:
    """
    Clase que crea una matriz de `Node` y añade dos nodos virtuales.
    conectados a los extremos
    """

    def __init__(self, size: int, player_id: int) -> None:
        self.size = size
        self.player = player_id
        self.matrix: List[List[Node]] = []
        self.free_cells: Set[Tuple[int, int]] = set()
        self.player_cells: Set[Tuple[int, int]] = set()
        self.opp_cells: Set[Tuple[int, int]] = set()
        self.node_left: Optional[Node] = None
        self.node_right: Optional[Node] = None
        self.node_up: Optional[Node] = None
        self.node_bottom: Optional[Node] = None
        self.opp: Optional[int] = None
        self.center_dom_self = 0.0
        self.center_dom_opp = 0.0
        self.edges_dom_self = 0.0
        self.edges_dom_opp = 0.0
        self.matrix_center = None
        self.matrix_edges_self = None
        self.matrix_edges_opp = None
        self.create_node_matrix()
        self.move_counter = 0

    def get_dom(self, player_id: int):
        if player_id == self.player:
            return round(self.center_dom_self if self.move_counter <= Minimax.ctrl_board else self.edges_dom_self, 2)
        
        return round(self.center_dom_opp if self.move_counter <= Minimax.ctrl_board else self.edges_dom_opp, 2)
    
    def detect_opponent_move(self, board: HexBoard) -> None:
        """
        Recibe un `HexBoard` y devuelve la coordenada (row, col) de una nueva 
        ficha puesta por el adversario 
        """
        
        for (r, c) in self.free_cells:
            if board.board[r][c] != 0:
                self.mark_node_at(r, c, self.opp)
                return None
            
        return None

    def is_different_board(self, board: HexBoard, sample_size: int = 3) -> bool:
        """
        Select up to `sample_size` random cells from the union of
        `player_cells` and `opp_cells`. If at least one of the selected
        cells is free on `board` (i.e. `board.board[r][c] == 0`) return True.

        Returns False if there are no candidate cells or none of the sampled
        cells are free.
        """
        if board.size != self.size:
            return True
        
        # Combine player's and opponent's cells
        candidates = list(self.player_cells | self.opp_cells)
        if not candidates:
            return False

        sampled = random.sample(candidates, min(sample_size, len(candidates)))

        for (r, c) in sampled:
            if board.board[r][c] == 0:
                return True

        return False

    def is_cell_available(self, r: int, c: int) -> bool:
        """
        Devuelve True si existe (r, c) está desocupada
        """
        
        if not (0 <= r < self.size and 0 <= c < self.size):
            return False

        return (r, c) in self.free_cells

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

        matrix: List[List[Node]] = [[Node(r, c) for c in range(self.size)] for r in range(self.size)]
        # Guardar el id del jugador, pero no usarlo para decidir qué extremos
        # conectar: creamos los cuatro nodos extremos y los enlazamos todos.
        self.opp = 2 if self.player == 1 else 1
        self.matrix_center = [[0 for _ in range(self.size)] for _ in range(self.size)]
        self.matrix_edges_self = [[0 for _ in range(self.size)] for _ in range(self.size)]
        self.matrix_edges_opp = [[0 for _ in range(self.size)] for _ in range(self.size)]

        for r in range(self.size):
            for c in range(self.size):
                self.free_cells.add((r, c))
                
                # Territorial
                self.matrix_center[r][c] = self.territorial_control(r,c)
                self.matrix_edges_opp[r][c] = self.territorial_control(r,c,self.opp)
                self.matrix_edges_self[r][c] = self.territorial_control(r,c,self.player)
                
                # Neighbors 
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

    def mark_node_at(self, r: int, c: int, player_id: int = None, mark: bool = True) -> None:
        """
        Marca o desmarca el nodo en (r, c).

        - Si `player_id` es 1 o 2, guarda `player_id` en `marked`.
        - Si `player_id` es `None`, guarda 0 en `marked` (desmarca).
        - Lanza `IndexError` si la matriz no existe o las coordenadas están
          fuera de rango.
        - Lanza `ValueError` si la posición ya es `None` (nodo eliminado).
        """

        node = self.matrix[r][c]

        # Desmarcar (player_id is None)
        if not mark and node.marked != 0:
            
            self.move_counter -= 1
            # Quitar de los sets correspondientes
            if self.player == player_id:
                self.player_cells.discard((r, c))
                self.edges_dom_self -= self.matrix_edges_self[r][c]
                self.center_dom_self -= self.matrix_center[r][c]
                    
            else:
                self.opp_cells.discard((r, c))
                self.edges_dom_opp -= self.matrix_edges_opp[r][c]
                self.center_dom_opp -= self.matrix_center[r][c]  

            node.marked = 0
            # Añadir a free_cells
            self.free_cells.add((r, c))
            return
        elif node.marked == 0:
            # Si la casilla estaba libre, actualizar move_counter y quitar de free_cells
            self.move_counter += 1
            self.free_cells.discard((r, c))
            node.marked = player_id
            # Añadir a los sets correspondientes
            if player_id == self.player:
                self.player_cells.add((r, c))
                self.edges_dom_self += self.matrix_edges_self[r][c]
                self.center_dom_self += self.matrix_center[r][c]
            else:
                self.opp_cells.add((r, c))
                self.edges_dom_opp += self.matrix_edges_opp[r][c]
                self.center_dom_opp += self.matrix_center[r][c]
        
    def territorial_control(self, r: int, c: int, player: int = None) -> float:
        """
        Retorna valor entre 0 y 1.

        - Si `prioritize_borders` es False: mide cercania al centro.
        - Si `prioritize_borders` es True: mide cercania a los bordes
          relevantes para el jugador:
            - Jugador 1: izquierda-derecha.
            - Jugador 2: arriba-abajo.
        """
        
        if player is None:
            cr = (self.size - 1) // 2
            cc = (self.size - 1) // 2

            # Distancia euclidiana al centro.
            distancia = ((r - cr) ** 2 + (c - cc) ** 2) ** 0.5

            # Distancia maxima aproximada (esquinas).
            dist_max = ((0 - cr) ** 2 + (0 - cc) ** 2) ** 0.5
            if dist_max == 0:
                return 1.0

            # Normalizamos e invertimos (mas cerca -> valor mas alto).
            cercania = 1.0 - (distancia / dist_max)
            return round(max(0.0, min(1.0, cercania)),2)
        
        # Estrategia por bordes segun orientacion del jugador.
        if player == 1:
            # Jugador 1 conecta izquierda-derecha: importa la distancia al
            # borde izquierdo o derecho (la minima de ambas).
            nearest_border_dist = min(c, self.size - 1 - c)
        else:
            # Jugador 2 conecta arriba-abajo: importa la distancia al
            # borde superior o inferior (la minima de ambas).
            nearest_border_dist = min(r, self.size - 1 - r)

        max_nearest_dist = (self.size - 1) / 2
        if max_nearest_dist == 0:
            return 1.0

        cercania_borde = 1.0 - (nearest_border_dist / max_nearest_dist)
        return round(max(0.0, min(1.0, cercania_borde)), 2)

    def distance_between_extremes(self, player_id: int) -> Optional[Tuple[int, float]]:
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
        
        return max(distances[b], 0)

    def count_components(self, player: int) -> Tuple[int, int]:
        """
        Cuenta el número de componentes conexas y la cardinalidad de la
        componente más grande considerando solamente nodos de la matriz
        con `marked == player`.

        Retorna `(num_componentes, max_cardinalidad)`.
        """
        if not self.matrix:
            return 0, 0

        if player not in (1, 2):
            raise ValueError("player must be 1 or 2")

        n = self.size
        matrix = self.matrix
        visited = [0] * (n * n)
        components = 0
        max_card = 0

        # Seleccionar el conjunto de coordenadas a iterar según el jugador
        if player == self.player:
            coords_iter = self.player_cells
        else:
            coords_iter = self.opp_cells

        for r, c in coords_iter:
            idx = r * n + c
            if visited[idx]:
                continue

            start = matrix[r][c]
            if start.marked != player:
                continue

            # Nueva componente: DFS
            components += 1
            comp_id = components
            stack = [(r, c)]
            visited[idx] = 1
            start.id_comp = comp_id
            card = 0

            while stack:
                cr, cc = stack.pop()
                node = matrix[cr][cc]
                card += 1

                for neigh in node.neighbors:
                    nr, nc = neigh.r, neigh.c
                    # Ignorar extremos (coordenadas negativas)
                    if not (0 <= nr < n and 0 <= nc < n):
                        continue

                    nidx = nr * n + nc
                    if visited[nidx]:
                        continue

                    neigh_node = matrix[nr][nc]
                    if neigh_node.marked != player:
                        continue

                    visited[nidx] = 1
                    neigh_node.id_comp = comp_id
                    stack.append((nr, nc))

            if card > max_card:
                max_card = card

        return components, max_card

    def count_threatened_free_nodes(self, player: int, free_nodes: Optional[Iterable[Tuple[int, int]]] = None, comp_done: bool = False) -> int:
        """
        Cuenta nodos libres que se consideran amenazados respecto a `player`.

        Si se proporciona `free_nodes` (lista de coordenadas `(r, c)`),
        se iterará sólo sobre esa lista. Si no se
        proporciona, se recorre toda la matriz como antes.

        Un nodo libre se considera amenazado para `player` si entre sus
        vecinos hay al menos dos nodos marcados por `player` que pertenezcan
        a componentes conexas distintas (tienen `id_comp` distintos).
        """
        # Calcular id_comp sólo para el jugador indicado

        matrix = self.matrix
        n = self.size  
        
        if comp_done:
            self.count_components(player)

        threatened = 0

        if free_nodes is not None:
            for coords in free_nodes:
                rr, cc = coords
                node = matrix[rr][cc]
                    
                first_comp = None
                for neigh in node.neighbors:
                    nr, nc = neigh.r, neigh.c

                    if nr < 0 or nr >= n or nc < 0 or nc >= n:
                        continue

                    neigh_node = matrix[nr][nc]
                    if neigh_node.marked != player:
                        continue

                    comp = neigh_node.id_comp
                    if comp is None:
                        continue

                    if first_comp is None:
                        first_comp = comp
                    elif comp != first_comp:
                        threatened += 1
                        break

        return threatened

    def get_ordered_moves(self) -> list[Tuple[int, int]]:
        """
        Prioriza:
        1. Casillas adyacentes a fichas ya colocadas (propias o del rival)
        2. Dentro de cada grupo, ordena de mayor a menor según:
           - `self.matrix_center` si `self.move_counter <= Minimax.ctrl_board`
           - `self.matrix_edges_self` en caso contrario

        Devuelve una lista ordenada (adjacentes primero, luego las demás).
        """
        size = self.size

        # Escoger la matriz de valores a usar según el control del tablero
        use_center = self.move_counter <= Minimax.ctrl_board
        key_matrix = self.matrix_center if use_center else self.matrix_edges_self

        adjacent: list[Tuple[int, int]] = []
        others: list[Tuple[int, int]] = []

        for (r, c) in self.free_cells:
            is_adjacent = False
            for nr, nc in self._neighbors(r, c):
                if 0 <= nr < size and 0 <= nc < size:
                    if self.matrix[nr][nc].marked != 0:
                        is_adjacent = True
                        break

            if is_adjacent:
                adjacent.append((r, c))
            else:
                others.append((r, c))

        # Ordenar cada grupo de mayor a menor según la matriz seleccionada
        adjacent.sort(key=lambda rc: key_matrix[rc[0]][rc[1]], reverse=True)
        others.sort(key=lambda rc: key_matrix[rc[0]][rc[1]], reverse=True)

        # Adjacent primero, luego las demás
        return adjacent + others

class Minimax:
    """
    Contiene la heurística y el algoritmo minimax como métodos estáticos.
    """

    distance = 1       # distancia entre extremos
    components = 1     # numero de componentes
    max_component = 1  # cardinalidad de componente más grande
    threats = 1        # celdas amenazadas
    territory = 1      # dominio general sobre el tablero
    ctrl_board = 45    # factor de control territorial (activa el cambio de estrategia)
    
    @staticmethod
    def calculate_depth_simple(size: int, move_counter: int) -> int:
        value = size*size - move_counter
        
        if value <= 12:
            return 9
        elif value <= 14:
            return 7
        elif value <= 49:
            if size <= 6 or (size==7 and value > 28):
                return 5
        return 3
    
    @staticmethod
    def set_weights(*weights, graph: Optional["HexGraph"] = None) -> None:
        """
        Set weights for the heuristic.

        Behavior:
        - If `graph` is provided and `graph.move_counter <= 2`, chooses a
          random preset from three predefined weight vectors.
        - Otherwise uses the weights passed as positional args (either six
          separate numbers or a single iterable of six numbers).
        """
        presets = [
            # Muy agresivo en conexión (prioridad máxima a distance + threats)
            [240, 10, 5, 150, 60, 30],

            # Agresivo equilibrado (distance muy alto, threats fuerte)
            [220, 15, 8, 140, 70, 45],

            # Distance + threats fuerte, algo más de territorio
            [200, 20, 10, 130, 80, 50],

            # Punto medio sólido (buen balance para la mayoría de tableros)
            [180, 25, 12, 110, 90, 55],

            # Más territorial / influencia espacial (útil en tableros ≥11×11)
            [160, 30, 15, 100, 110, 70],

            # Distance + threats muy altos (estilo "Queenbee-like" clásico)
            [250, 5, 0, 180, 50, 25],

            # Control central fuerte (si ctrl_board mide bien bordes/centro)
            [170, 20, 10, 120, 85, 90],

            # Versión conservadora (más threats para defender, menos riesgo)
            [190, 15, 5, 160, 75, 40],

            # Alta distancia + amenazas, bajo todo lo demás
            [230, 8, 3, 170, 55, 35],

            # Equilibrado moderno (inspirado en bots 2020s con algo de ctrl)
            [210, 18, 10, 125, 95, 60],
        ]

        # If graph provided and early game (<= 2 moves), pick a random preset.
        if graph.move_counter <= 2:
            weights = random.choice(presets)

        Minimax.distance, Minimax.components, Minimax.max_component, Minimax.threats, Minimax.territory, Minimax.ctrl_board = weights
        Minimax.ctrl_board = math.floor((Minimax.ctrl_board / 100) * graph.size * graph.size)
    
    @staticmethod
    def calculate_heuristic(graph: HexGraph, free_node: Optional[Iterable[Tuple[int, int]]] = None) -> Optional[int]:

        dist_self  = graph.distance_between_extremes(graph.player)
        dist_opp = graph.distance_between_extremes(graph.opp)
            
        comp_num_self, max_card_self = graph.count_components(graph.player)
        comp_num_opp, max_card_opp = graph.count_components(graph.opp) 
        
        threat_cells_self = graph.count_threatened_free_nodes(graph.player, free_node)
        threat_cells_opp = graph.count_threatened_free_nodes(graph.opp, free_node)
        
        board_dom_self = graph.get_dom(graph.player)
        board_dom_opp = graph.get_dom(graph.opp)
        
        if dist_self is None:
            return -10000
        if dist_opp is None:
            return 10000

        return Minimax.distance*(dist_opp - dist_self) + Minimax.components*(comp_num_opp - comp_num_self) + Minimax.max_component*(max_card_self - max_card_opp) + Minimax.threats*(threat_cells_opp - threat_cells_self) + Minimax.territory*(board_dom_self - board_dom_opp)

    @staticmethod
    def preminimax(graph: "HexGraph", board: HexBoard) -> Optional[Tuple[int, int]]:
        """
        Selecciona una profundidad adecuada en función del tamaño del grafo
        y del momento, llama a `minimax`. Retorna el (valor, mejor_jugada).
        """
        # Detectar movimiento del oponente
        graph.detect_opponent_move(board)  

        # Ajustar pesos según el torneo (pasamos el grafo para decidir presets)
        Minimax.set_weights(16.9283, 2.8124, 5.1506, 16.7625, 1, 43.4143, graph=graph)
        
        size = graph.size
        
        deep = Minimax.calculate_depth_simple(size, graph.move_counter)

        _, best_move = Minimax.minimax(
            turno=0,
            profundidad=deep,
            graph=graph,
        )
        
        if best_move is not None:
            graph.mark_node_at(*best_move, graph.player)
            return best_move
        
        return None

    @staticmethod
    def minimax(
        turno: int,
        profundidad: int,
        graph: HexGraph,
        alpha: int = -sys.maxsize - 1,
        beta: int = sys.maxsize,
        maximizing: bool = True,
        moves: Optional[list[Tuple[int, int]]] = None,
    ) -> Tuple[float, Optional[Tuple[int, int]]]:
        """
        Minimax. Retorna (valor, mejor_jugada, promedio_de_hijos_directos).
        """
        if moves is None:
            moves = graph.get_ordered_moves()

        if turno >= profundidad - 1 or not moves:
            val = Minimax.calculate_heuristic(graph, moves)
            leaf_val = 0.0 if val is None else float(val)
            return leaf_val, None

        best_move = None

        if maximizing:
            max_eval = -sys.maxsize - 1

            for idx in range(len(moves)):
                r, c = moves[idx]

                # Marcar la jugada
                graph.mark_node_at(r, c, graph.player)

                # Crear la lista de movimientos restante sin mutar la original
                remaining_moves = moves[:idx] + moves[idx + 1 :]

                eval, _ = Minimax.minimax(
                    turno + 1,
                    profundidad,
                    graph,
                    alpha,
                    beta,
                    False,
                    remaining_moves,
                )

                # Desmarcar después de evaluar
                graph.mark_node_at(r, c, graph.player, mark=False)

                if eval > max_eval:
                    max_eval = eval
                    if turno == 0:
                        best_move = (r, c)

                alpha = max(alpha, eval)
                if beta <= alpha:
                    break

            return float(max_eval), best_move

        else:
            min_eval = sys.maxsize

            for idx in range(len(moves)):
                r, c = moves[idx]

                # Marcar la jugada del oponente
                graph.mark_node_at(r, c, graph.opp)

                remaining_moves = moves[:idx] + moves[idx + 1 :]

                eval, _ = Minimax.minimax(
                    turno + 1,
                    profundidad,
                    graph,
                    alpha,
                    beta,
                    True,
                    remaining_moves,
                )

                # Desmarcar después de evaluar
                graph.mark_node_at(r, c, graph.opp, mark=False)

                if eval < min_eval:
                    min_eval = eval

                beta = min(beta, eval)
                if beta <= alpha:
                    break

            return float(min_eval), best_move

# Second Strategy
#-----------------------------------------------------------------------
