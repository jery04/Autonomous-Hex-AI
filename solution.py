from player import Player
from board import HexBoard
from typing import List, Tuple, Optional, Set, Iterable
import sys
import math
import random
import statistics


class SmartPlayer(Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph: Optional[HexGraph] = None

    def update_graphs(self, board: HexBoard) -> None:
        """
        Crea lel grarfo si no existe y detecta la última jugada del oponente
        """
        # Reuse existing graphs if present; create them if not.
        if self.graph is None:
            self.graph = HexGraph(size=board.size, player_id=self.player_id)
        
    def play(self, board: HexBoard) -> tuple:

        best_move = None
        if board.size <= 7:
            # Sincronizar y actualizar grafos a partir del tablero
            self.update_graphs(board) 
            
            # Usar preminimax para elegir profundidad según tamaño
            return Minimax.preminimax(self.graph, board)
        else:
            print("Soon...")
        
        # Algo raro
        return (-1, -1)

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
    Clase que crea una matriz de `Node` y añade dos nodos extremos.
    - `create_node_matrix(size, orientation)` crea la matriz y enlaza vecinos.
    - `orientation` 1 = extremos izquierda-derecha, 2 = extremos arriba-abajo.
    Los nodos extremos se exponen en `extreme1` y `extreme2`.
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
        self.create_node_matrix()
        self.move_counter = 0

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

        for r in range(self.size):
            for c in range(self.size):
                self.free_cells.add((r, c))
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

        node = self.matrix[r][c]

        # Desmarcar (player_id is None)
        if player_id is None:
            # Solo decrementar contador si antes estaba marcado.
            if node.marked != 0:
                self.move_counter -= 1
                # Quitar de los sets correspondientes
                if node.marked == self.player:
                    self.player_cells.discard((r, c))
                elif node.marked == self.opp:
                    self.opp_cells.discard((r, c))

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
            elif player_id == self.opp:
                self.opp_cells.add((r, c))
        
    def territorial_control(self, r: int, c: int, player: int = None) -> float:
        """
        Retorna valor entre 0 y 1.

        - Si `prioritize_borders` es False: mide cercania al centro.
        - Si `prioritize_borders` es True: mide cercania a los bordes
          relevantes para el jugador:
            - Jugador 1: izquierda-derecha.
            - Jugador 2: arriba-abajo.
        """
        
        if self.move_counter <  math.floor(self.size * self.size * Minimax.ctrl_board / 100):
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
            return max(0.0, min(1.0, cercania))

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
        return max(0.0, min(1.0, cercania_borde))

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

        from collections import deque

        position = 0.0
        # Evita contar dos veces el mismo nodo propio en distintas rutas/relajaciones.
        counted_nodes: Set[Node] = set()
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

                if neigh_mark == player_id and neigh not in counted_nodes:
                    if 0 <= neigh.r < self.size and 0 <= neigh.c < self.size:
                        position += self.territorial_control(neigh.r, neigh.c, player_id)
                    counted_nodes.add(neigh)

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
            return None, position
        
        result = max(distances[b], 0)
        return result, position

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
        if not self.matrix:
            return 0

        if player not in (1, 2):
            raise ValueError("player must be 1 or 2")

        matrix = self.matrix
        n = self.size
        # Calcular id_comp sólo para el jugador indicado
        
        if comp_done:
            self.count_components(player)

        threatened = 0

        # Recorrido completo (hot path): evitar generadores y excepciones.
        if free_nodes is None:
            for r in range(n):
                row = matrix[r]
                for c in range(n):
                    node = row[c]
                    if node.marked != 0:
                        continue

                    first_comp = None
                    for neigh in node.neighbors:
                        nr, nc = neigh.r, neigh.c

                        # Ignorar extremos y cualquier vecino fuera de matriz.
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
        else:
            for coords in free_nodes:
                # Validar entrada de forma barata: se esperan tuplas (r, c).
                if not isinstance(coords, (tuple, list)) or len(coords) != 2:
                    continue

                rr, cc = coords
                if rr < 0 or rr >= n or cc < 0 or cc >= n:
                    continue

                node = matrix[rr][cc]
                if node.marked != 0:
                    continue

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

class Minimax:
    """
    Contiene la heurística y el algoritmo minimax como métodos estáticos.
    """

    distance = 130.9652      # distancia entre extremos
    components = 21.5188       # numero de componentes
    max_component = 19.7512      # cardinalidad de componente más grande
    threats = 120.2703      # celdas amenazadas
    territory = 73.6211     # dominio general sobre el tablero
    ctrl_board = 31.5804      # factor de control territorial (celdas cercanas al centro o bordes relevantes)
    
    @staticmethod
    def set_weights(*weights) -> None:
        """
        Set weights for the heuristic. Accepts either a single iterable
        (list/tuple) of six numbers or six numeric positional arguments.
        """
        # Unpack if a single iterable was passed
        if len(weights) == 1 and isinstance(weights[0], (list, tuple)):
            vals = list(weights[0])
        else:
            vals = list(weights)

        if len(vals) != 6:
            raise ValueError("set_weights expects 6 weight values")

        Minimax.distance, Minimax.components, Minimax.max_component, Minimax.threats, Minimax.territory, Minimax.ctrl_board = vals
    
    @staticmethod
    def calculate_heuristic(graph: HexGraph, free_node: Optional[Iterable[Tuple[int, int]]] = None) -> Optional[int]:
        if graph is None:
            return None

        if graph.player not in (1, 2) or graph.opp not in (1, 2):
            return None

        dist_self, board_dom_self  = graph.distance_between_extremes(graph.player)
        dist_opp, board_dom_opp = graph.distance_between_extremes(graph.opp)
        
        comp_num_self, max_card_self = graph.count_components(graph.player)
        comp_num_opp, max_card_opp = graph.count_components(graph.opp) 
        
        threat_cells_self = graph.count_threatened_free_nodes(graph.player, free_node)
        threat_cells_opp = graph.count_threatened_free_nodes(graph.opp, free_node)
        
        if dist_self is None:
            return -10000
        if dist_opp is None:
            return 10000

        return Minimax.distance*(dist_opp - dist_self) + Minimax.components*(comp_num_opp - comp_num_self) + Minimax.max_component*(max_card_self - max_card_opp) + Minimax.threats*(threat_cells_opp - threat_cells_self) + Minimax.territory*(board_dom_self - board_dom_opp)

    @staticmethod
    def preminimax(graph: "HexGraph", board: HexBoard) -> Optional[Tuple[int, int]]:
        """
        Selecciona una profundidad adecuada en función del tamaño del grafo
        y llama a `minimax`. Retorna el (valor, mejor_jugada).

        Reglas de profundidad:
        - size <= 3  -> profundidad 11
        - 4 <= size <=5 -> profundidad 5
        - 6 <= size <=7 -> profundidad 3
        - por defecto -> profundidad 3
        """
        
        # Sincronizar grafos con el tablero actual
        graph.detect_opponent_move(board)  

        size = getattr(graph, "size", None)
        
        if size <= 3:
            profundidad = 11
            #a = 1, b=2, c=3, d=4, e=5, f=6
        elif 4 <= size <= 5:
            profundidad = 5
            #164.8978, 7.6193, 31.2489, 89.6845, 66.9143, 27.9527
            #169.5289, 4.0, 30.9841, 78.4032, 86.1154, 30.2524
            #130.9652, 21.5188, 19.7512, 120.2703, 73.6211, 31.5804
            #125.7379, 19.3651, 16.104, 114.8725, 72.9301, 33.5532
            Minimax.set_weights(169.5289, 4.0, 30.9841, 78.4032, 86.1154, 30.2524)
        elif 6 <= size <= 7:
            profundidad = 3
            #a = 1, b=2, c=3, d=4, e=5, f=6
            

        _, best_move = Minimax.minimax(
            turno=0,
            profundidad=profundidad,
            graph=graph,
        )
        
        if best_move is not None:
            graph.mark_node_at(*best_move, graph.player)
            return best_move
        
        return None

    @staticmethod
    def get_ordered_moves(graph: HexGraph) -> Set[Tuple[int, int]]:
        """
        Prioriza:
        1. Casillas adyacentes a fichas ya colocadas (propias o del rival)
        2. Casillas cercanas al centro
        """
        size = graph.size
        center_r, center_c = size // 2, size // 2
        max_dist = 2 * (size - 1)

        candidates_by_dist: list[list[tuple[int, int]]] = [[] for _ in range(max_dist + 1)]
        others_by_dist: list[list[tuple[int, int]]] = [[] for _ in range(max_dist + 1)]

        for (r, c) in graph.free_cells:

            is_adjacent = False
            for nr, nc in graph._neighbors(r, c):
                if 0 <= nr < size and 0 <= nc < size:
                    if graph.matrix[nr][nc].marked != 0:
                        is_adjacent = True
                        break

            dist_center = abs(r - center_r) + abs(c - center_c)

            if is_adjacent:
                candidates_by_dist[dist_center].append((r, c))
            else:
                others_by_dist[dist_center].append((r, c))

        ordered_moves: list[tuple[int, int]] = []

        for dist in range(max_dist + 1):
            ordered_moves.extend(candidates_by_dist[dist])

        for dist in range(max_dist + 1):
            ordered_moves.extend(others_by_dist[dist])

        # Primero todos los que tocan algo, luego el resto.
        return set(ordered_moves)

    @staticmethod
    def minimax(
        turno: int,
        profundidad: int,
        graph: HexGraph,
        alpha: int = -sys.maxsize - 1,
        beta: int = sys.maxsize,
        maximizing: bool = True,
        moves: Optional[Set[Tuple[int, int]]] = None,
    ) -> Tuple[float, Optional[Tuple[int, int]], float]:
        """
        Minimax. Retorna (valor, mejor_jugada, promedio_de_hijos_directos).
        """
        if moves is None:
            moves = Minimax.get_ordered_moves(graph)
            
        if turno >= profundidad-1 or not moves:
            val = Minimax.calculate_heuristic(graph, moves)
            leaf_val = 0.0 if val is None else float(val)
            return leaf_val, None

        best_move = None

        if maximizing:
            max_eval = -sys.maxsize - 1

            for r, c in moves:
                
                # Marcar la jugada
                graph.mark_node_at(r, c, graph.player)
                # Remover la casilla libre
                moves.remove((r, c))
               
                eval, _ = Minimax.minimax(
                    turno + 1,
                    profundidad,
                    graph,
                    alpha,
                    beta,
                    False,
                    moves,
                )
                
                # Volver a agregar la casilla libre para el siguiente hijo
                moves.add((r, c))
                
                # Desmarcar después de evaluar
                graph.mark_node_at(r, c, None)  
                
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
            
            for r, c in moves:

                # Marcar la jugada
                graph.mark_node_at(r, c, graph.opp)

                #remaining_moves = moves - {(r, c)}
                moves.remove((r, c))

                eval, _ = Minimax.minimax(
                    turno + 1,
                    profundidad,
                    graph,
                    alpha,
                    beta,
                    True,
                    moves,
                )

                # Volver a agregar la casilla libre para el siguiente hijo
                moves.add((r, c))
                
                # Desmarcar después de evaluar
                graph.mark_node_at(r, c, None)

                if eval < min_eval:
                    min_eval = eval

                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return float(min_eval), best_move
