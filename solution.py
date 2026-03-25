from player import Player
from board import HexBoard
from typing import List, Tuple, Optional, Set, Iterable
import sys
import math
import random
from collections import deque
import time

# LLamada Principal
#-----------------------------------------------------------------------
class SmartPlayer(Player):
    """Adaptive Hex player that switches strategy by board size.
    It uses minimax on small boards and MCTS on larger ones."""

    def __init__(self, *args, **kwargs):
        """Initialize the smart player and strategy caches.
        It starts with no graph or MCTS tree loaded."""
        super().__init__(*args, **kwargs)
        self.graph: Optional[HexGraph] = None
        self.mcts: Optional[MCTS] = None

    def play(self, board: HexBoard) -> tuple:
        """Pick the next move for the current board state.
        The method synchronizes internal state and delegates to the selected search strategy."""

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
    """Graph node representing one Hex cell or a virtual border endpoint.
    It stores coordinates, neighbors, ownership mark, and component id."""

    def __init__(self, r: int, c: int, marked: int = 0):
        """Create a node with coordinates and an optional owner mark.
        New nodes start with an empty neighbor list and no component id."""
        self.r = r
        self.c = c
        self.neighbors: List["Node"] = []  # lista de Nodos adyacentes
        self.marked = marked
        self.id_comp: Optional[int] = None  # id de componente conexa (player, comp_id)

    def __repr__(self) -> str:
        """Return a compact debug representation of the node.
        The string includes only row and column coordinates."""
        return f"Node({self.r},{self.c})"

class HexGraph:
    """Hex board graph with cell nodes and virtual border nodes.
    It tracks occupancy sets and utility data used by the heuristic evaluator."""

    def __init__(self, size: int, player_id: int) -> None:
        """Build graph containers and initialize board-dependent structures.
        The constructor creates the node matrix and resets tracking counters."""
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
        self.last_move_opp = (-1,-1)
        self.last_move_own = (-1,-1)

    def get_dom(self, player_id: int):
        """Return rounded territorial dominance for the requested player.
        The metric switches between center and edge weighting by game phase."""
        if player_id == self.player:
            return round(self.center_dom_self if self.move_counter <= Minimax.ctrl_board else self.edges_dom_self, 2)
        
        return round(self.center_dom_opp if self.move_counter <= Minimax.ctrl_board else self.edges_dom_opp, 2)
    
    def detect_opponent_move(self, board: HexBoard) -> None:
        """Update internal state with the opponent's latest move.
        It scans free cells and marks the first position that became occupied."""
        
        for (r, c) in self.free_cells:
            if board.board[r][c] != 0:
                self.mark_node_at(r, c, self.opp)
                self.last_move_opp = (r, c)
                return None
            
        return None

    def is_different_board(self, board: HexBoard, sample_size: int = 3) -> bool:
        """Heuristically detect whether the external board diverged from this graph.
        It samples known occupied cells and returns True if any sample became empty."""
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
        """Check if a coordinate is in bounds and currently free.
        Returns False for out-of-range coordinates or occupied cells."""
        
        if not (0 <= r < self.size and 0 <= c < self.size):
            return False

        return (r, c) in self.free_cells

    def _neighbors(self, r: int, c: int) -> List[Tuple[int, int]]:
        """Return valid neighbor coordinates for a hex cell.
        The offset pattern depends on row parity."""
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
        """Create all board nodes, adjacency lists, and virtual border links.
        It also precomputes territorial score matrices and resets occupancy sets."""
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
        """Mark or unmark a board node and update derived bookkeeping.
        This method keeps move count, free cells, ownership sets, and dominance totals in sync."""

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
        """Compute a normalized positional score in the range [0, 1].
        It evaluates center proximity by default or goal-border proximity for a given player."""
        
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
        """Estimate connection distance between the player's two goal borders using 0-1 BFS.
        Own stones have zero traversal cost and empty cells have unit cost."""
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
        """Count connected components for one player's stones.
        It returns both the number of components and the size of the largest one."""
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
        """Count free cells threatened by a player via multi-component adjacency.
        A cell is threatened when it touches at least two distinct connected components of that player."""
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
        """Return legal moves ordered by tactical proximity and positional value.
        Adjacent free cells come first, then both groups are sorted by the active scoring matrix."""
        size = self.size
        
        # Escoger la matriz de valores a usar según el control del tablero
        use_center = self.move_counter <= Minimax.ctrl_board
        key_matrix = self.matrix_center if use_center else self.matrix_edges_self

        adjacent: list[Tuple[int, int]] = []
        others: list[Tuple[int, int]] = []
        priority: list[Tuple[int, int]] = []

        for (r, c) in self.free_cells:
            is_adjacent = False
            is_priority = False
            for nr, nc in self._neighbors(r, c):
                if (0 <= nr < size and 0 <= nc < size) and self.matrix[nr][nc].marked != 0:
                    if (nr,nc) == self.last_move_opp or (nr,nc) == self.last_move_own:
                        is_priority = True
                        break
                    is_adjacent = True
                    break
                        
            if is_priority:
                priority.append((r, c))
            elif is_adjacent:
                adjacent.append((r, c))
            else:
                others.append((r, c))

        # Ordenar cada grupo de mayor a menor según la matriz seleccionada
        adjacent.sort(key=lambda rc: key_matrix[rc[0]][rc[1]], reverse=True)
        others.sort(key=lambda rc: key_matrix[rc[0]][rc[1]], reverse=True)
        priority.sort(key=lambda rc: key_matrix[rc[0]][rc[1]], reverse=True)

        # Adjacent primero, luego las demás
        return priority + adjacent + others

class Minimax:
    """Static minimax engine and heuristic configuration.
    It provides depth selection, weight management, and alpha-beta search."""

    distance = 1       # distancia entre extremos
    components = 1     # numero de componentes
    max_component = 1  # cardinalidad de componente más grande
    threats = 1        # celdas amenazadas
    territory = 1      # dominio general sobre el tablero
    ctrl_board = 45    # factor de control territorial (activa el cambio de estrategia)

    @staticmethod
    def calculate_depth_simple(size: int, move_counter: int) -> int:
        """Choose a search depth from board size and remaining cells.
        The policy increases depth in late game positions."""
        value = size*size - move_counter
        
        if value <= 12:
            return 9
        elif value <= 14:
            return 7
        elif value <= 49:
            if size <= 6 or (size==7 and value > 42):
                return 5
        return 3
    
    @staticmethod
    def set_weights(*weights, graph: Optional["HexGraph"] = None) -> None:
        """Configure heuristic weights and phase threshold for minimax.
        In early moves it may choose a random preset to diversify openings."""
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
        """Evaluate a position using distance, connectivity, threats, and territory terms.
        Higher values favor the current player and terminal disconnections get extreme scores."""

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
        """Prepare graph state and run minimax with adaptive depth.
        The chosen move is committed to the internal graph before returning it."""
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
            graph.last_move_own = best_move
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
        """Run recursive alpha-beta minimax over ordered legal moves.
        It returns the evaluated score and best root move when available."""
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
class DisjointSet:
    """Disjoint-set union structure for connectivity checks.
    It supports near-constant-time find and union operations."""

    def __init__(self, n: int) -> None:
        """Initialize n singleton sets.
        Parents start as self references and all ranks start at zero."""
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """Find the representative of x with path compression.
        Compression flattens paths to speed up future queries."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: int, b: int) -> None:
        """Merge the sets that contain a and b.
        Union-by-rank keeps trees shallow for efficient lookups."""
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        else:
            self.parent[rb] = ra
            if self.rank[ra] == self.rank[rb]:
                self.rank[ra] += 1

class MCTSNode:
    """Tree node used by Monte Carlo Tree Search.
    It stores move metadata, statistics, and expansion state."""

    def __init__(
        self,
        move: Optional[Tuple[int, int]] = None,
        parent: Optional["MCTSNode"] = None,
        player_just_moved: Optional[int] = None,
    ) -> None:
        """Create a search node for a move and its parent context.
        Statistics and child containers are initialized empty."""
        self.move = move
        self.parent = parent
        self.player_just_moved = player_just_moved
        self.children: List["MCTSNode"] = []
        self.untried_moves: Optional[List[Tuple[int, int]]] = None
        self.visits = 0
        self.wins = 0.0

    def uct_select_child(self, exploration: float) -> "MCTSNode":
        """Select the child with the highest UCT score.
        Unvisited children are treated as infinitely attractive."""
        log_parent = math.log(self.visits)

        def uct_value(child: "MCTSNode") -> float:
            if child.visits == 0:
                return float("inf")
            return (child.wins / child.visits) + exploration * math.sqrt(log_parent / child.visits)

        return max(self.children, key=uct_value)

    def add_child(self, move: Tuple[int, int], player_just_moved: int) -> "MCTSNode":
        """Create and append a child node for the given move.
        The move is removed from the untried list when present."""
        child = MCTSNode(move=move, parent=self, player_just_moved=player_just_moved)
        self.children.append(child)
        if self.untried_moves is not None:
            self.untried_moves.remove(move)
        return child

    def update(self, winner: int) -> None:
        """Backpropagate one simulation result into this node.
        Visits always increase, and wins increase only for matching players."""
        self.visits += 1
        if self.player_just_moved is not None and winner == self.player_just_moved:
            self.wins += 1.0

class MCTS:
    """Monte Carlo Tree Search player for larger Hex boards.
    It runs time-bounded simulations and returns the most visited legal move."""

    exploration = 1.41421356237

    def __init__(self, size: int, player_id: int) -> None:
        """Initialize MCTS state, caches, and board snapshots.
        The instance stores tracked cell sets and DSU caches for rollouts."""
        self.root: Optional[MCTSNode] = None
        self.root_player: Optional[int] = None
        self.player: Optional[int] = player_id
        self.opp: Optional[int] = self._other(player_id)

        self._last_my_cells: Set[Tuple[int, int]] = set()
        self._last_opp_cells: Set[Tuple[int, int]] = set()
        self._last_free_cells: Set[Tuple[int, int]] = {(r, c) for r in range(size) for c in range(size)}
        self._last_size: int = size

        self._dsu_cache: dict = {}

    @staticmethod
    def _other(player_id: int) -> int:
        """Return the opponent id for a two-player game.
        Valid ids are expected to be 1 and 2."""
        return 3 - player_id

    @staticmethod
    def _new_root(player_just_moved: int) -> MCTSNode:
        """Create a fresh root node for the current board state.
        The root has no move and no parent."""
        return MCTSNode(move=None, parent=None, player_just_moved=player_just_moved)

    def _sync_state_from_board(self, board: HexBoard) -> None:
        """Rebuild tracked cell sets from the external board.
        This also clears cached DSU structures tied to previous states."""
        self._last_size = board.size
        self._last_my_cells = set()
        self._last_opp_cells = set()
        self._last_free_cells = set()

        for r in range(board.size):
            for c in range(board.size):
                value = board.board[r][c]
                if value == 0:
                    self._last_free_cells.add((r, c))
                elif value == self.player:
                    self._last_my_cells.add((r, c))
                elif value == self.opp:
                    self._last_opp_cells.add((r, c))

        self._dsu_cache.clear()

    def _commit_my_move(self, move: Tuple[int, int]) -> Tuple[int, int]:
        """Persist the selected move in tracked local sets.
        The move is marked as owned and removed from free cells."""
        self._last_my_cells.add(move)
        #self._last_opp_cells.discard(move)
        self._last_free_cells.discard(move)
        return move

    def is_different_board(self, board: HexBoard, sample_size: int = 3) -> bool:
        """Heuristically check whether the board diverged from tracked history.
        It samples known occupied cells and flags differences if any become empty."""
        if board.size != self._last_size:
            return True

        candidates = list(self._last_my_cells | self._last_opp_cells)
        if not candidates:
            return False

        sampled = random.sample(candidates, min(sample_size, len(candidates)))
        return any(board.board[r][c] == 0 for (r, c) in sampled)
    
    def _neighbors(self, r: int, c: int) -> List[Tuple[int, int]]:
        """Return in-bounds hex neighbors for a coordinate.
        Neighbor offsets vary with row parity."""
        size = self._last_size
        deltas = [(-1, -1), (-1, 0), (0, 1), (1, 0), (1, -1), (0, -1)] if r % 2 != 0 else [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (0, -1)]
        candidates = [(r + dr, c + dc) for dr, dc in deltas]
        return [(nr, nc) for nr, nc in candidates if 0 <= nr < size and 0 <= nc < size]

    def _ordered_moves(self) -> List[Tuple[int, int]]:
        """List free moves with adjacency priority.
        Cells next to any occupied position are returned before the rest."""
        occupied = self._last_my_cells | self._last_opp_cells
        adjacent_free: Set[Tuple[int, int]] = set()
        for r, c in occupied:
            for n in self._neighbors(r, c):
                if n in self._last_free_cells:
                    adjacent_free.add(n)
        return list(adjacent_free) + list(self._last_free_cells - adjacent_free)

    def _mark_at(
        self,
        board_state: List[List[int]],
        r: int,
        c: int,
        player_id: Optional[int],
    ) -> None:
        """Apply or revert a move on a simulation board.
        Related DSU caches and free-cell tracking are updated consistently."""
        state_id = id(board_state)
        self._dsu_cache.pop((state_id, 1), None)
        self._dsu_cache.pop((state_id, 2), None)

        if player_id is None:
            board_state[r][c] = 0
            self._last_free_cells.add((r, c))
            return

        board_state[r][c] = player_id
        self._last_free_cells.discard((r, c))

    def _build_or_get_dsu(self, board_state: List[List[int]], player_id: int) -> DisjointSet:
        """Build or reuse a DSU connectivity view for a board state and player.
        The cache key is based on board identity and player id."""
        key = (id(board_state), player_id)
        cached = self._dsu_cache.get(key)
        if cached is not None and cached[1] == self._last_size:
            return cached[0]

        dsu = DisjointSet(self._last_size * self._last_size)

        def idx(rr: int, cc: int) -> int:
            return rr * self._last_size + cc

        for r in range(self._last_size):
            for c in range(self._last_size):
                if board_state[r][c] != player_id:
                    continue
                for nr, nc in self._neighbors(r, c):
                    if board_state[nr][nc] == player_id:
                        dsu.union(idx(r, c), idx(nr, nc))

        self._dsu_cache[key] = (dsu, self._last_size)
        return dsu

    def _has_connection(self, board_state: List[List[int]], player_id: int) -> bool:
        """Check whether a player has connected their two goal borders.
        Connectivity is tested using DSU representatives on border stones."""
        if self._last_size == 0 or player_id not in (1, 2):
            return False

        dsu = self._build_or_get_dsu(board_state, player_id)

        def idx(rr: int, cc: int) -> int:
            return rr * self._last_size + cc

        if player_id == 1:
            starts = [idx(r, 0) for r in range(self._last_size) if board_state[r][0] == player_id]
            ends = [idx(r, self._last_size - 1) for r in range(self._last_size) if board_state[r][self._last_size - 1] == player_id]
        else:
            starts = [idx(0, c) for c in range(self._last_size) if board_state[0][c] == player_id]
            ends = [idx(self._last_size - 1, c) for c in range(self._last_size) if board_state[self._last_size - 1][c] == player_id]

        if not starts or not ends:
            return False

        end_roots = {dsu.find(x) for x in ends}
        return any(dsu.find(x) in end_roots for x in starts)

    def _rollout(
        self,
        board_state: List[List[int]],
        player_to_move: int,
        played_moves: List[Tuple[int, int]],
    ) -> int:
        """Run a playout from the current simulation state.
        Moves are applied in ordered sequence until one player connects."""
        available = self._ordered_moves()

        current = player_to_move
        for r, c in available:
            self._mark_at(board_state, r, c, current)
            played_moves.append((r, c))
            if self._has_connection(board_state, current):
                return current
            current = self._other(current)

        return self._other(player_to_move)

    def best_move(self, board: HexBoard) -> Optional[Tuple[int, int]]:
        """Search for the best move using time-bounded MCTS iterations.
        The final choice is the most visited legal child from the root."""

        self._sync_state_from_board(board)

        if not self._last_free_cells:
            return None

        # Rebuild the search tree from the current legal position to avoid stale branches.
        self.root = self._new_root(self.opp)
        self.root_player = self.player

        root = self.root
        sim_board = [row[:] for row in board.board]

        time_limit = 3.1
        start_time = time.perf_counter()

        while time.perf_counter() - start_time < time_limit:
            node = root
            current_player = self.player
            played_moves: List[Tuple[int, int]] = []
            winner: Optional[int] = None

            if node.untried_moves is None:
                node.untried_moves = self._ordered_moves()

            while node.untried_moves == [] and node.children:
                node = node.uct_select_child(MCTS.exploration)
                r, c = node.move
                self._mark_at(sim_board, r, c, current_player)
                played_moves.append((r, c))

                if self._has_connection(sim_board, current_player):
                    winner = current_player
                    break

                current_player = self._other(current_player)
                if node.untried_moves is None:
                    node.untried_moves = self._ordered_moves()

            if winner is None and node.untried_moves:
                move = random.choice(node.untried_moves)
                self._mark_at(sim_board, move[0], move[1], current_player)
                played_moves.append(move)
                node = node.add_child(move, current_player)

                if self._has_connection(sim_board, current_player):
                    winner = current_player
                else:
                    current_player = self._other(current_player)

            if winner is None:
                winner = self._rollout(sim_board, current_player, played_moves)

            while node is not None:
                node.update(winner)
                node = node.parent

            while played_moves:
                r, c = played_moves.pop()
                self._mark_at(sim_board, r, c, None)

        if not root.children:
            move = random.choice(tuple(self._last_free_cells))
            return self._commit_my_move(move)

        legal_children = [
            child
            for child in root.children
            if child.move in self._last_free_cells and board.board[child.move[0]][child.move[1]] == 0
        ]

        if not legal_children:
            move = random.choice(tuple(self._last_free_cells))
            return self._commit_my_move(move)

        best_child = max(legal_children, key=lambda child: child.visits)
        self.root = best_child
        self.root.parent = None
        self.root_player = self.opp
        
        return self._commit_my_move(best_child.move)
