from typing import List, Optional, Set, Tuple
import math
import random
import time
from board import HexBoard


class DisjointSet:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        # Path compression
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        # Union by rank
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        else:
            self.parent[rb] = ra
            if self.rank[ra] == self.rank[rb]:
                self.rank[ra] += 1

class MCTSNode:
    def __init__(
        self,
        move: Optional[Tuple[int, int]] = None,
        parent: Optional["MCTSNode"] = None,
        player_just_moved: Optional[int] = None,
    ) -> None:
        self.move = move
        self.parent = parent
        self.player_just_moved = player_just_moved
        self.children: List["MCTSNode"] = []
        self.untried_moves: Optional[List[Tuple[int, int]]] = None
        self.visits = 0
        self.wins = 0.0

    def uct_select_child(self, exploration: float) -> "MCTSNode":
        log_parent = math.log(self.visits)

        def uct_value(child: "MCTSNode") -> float:
            if child.visits == 0:
                return float("inf")
            exploit = child.wins / child.visits
            explore = exploration * math.sqrt(log_parent / child.visits)
            return exploit + explore

        return max(self.children, key=uct_value)

    def add_child(self, move: Tuple[int, int], player_just_moved: int) -> "MCTSNode":
        child = MCTSNode(move=move, parent=self, player_just_moved=player_just_moved)
        self.children.append(child)
        if self.untried_moves is not None:
            self.untried_moves.remove(move)
        return child

    def update(self, winner: int) -> None:
        self.visits += 1
        if self.player_just_moved is not None and winner == self.player_just_moved:
            self.wins += 1.0

class MCTS:
    # Presupuesto razonable para GUI; escalar por tamaño de tablero.
    exploration = 1.41421356237

    def __init__(self) -> None:
        self.root: Optional[MCTSNode] = None
        self.root_player: Optional[int] = None
        self.player: Optional[int] = None
        self.opp: Optional[int] = None
        self._last_my_cells: Optional[Set[Tuple[int, int]]] = None
        self._last_opp_cells: Optional[Set[Tuple[int, int]]] = None
        self._last_free_cells: Optional[Set[Tuple[int, int]]] = None
        self._last_size: Optional[int] = None
        self._dsu_cache: dict = {}

    @staticmethod
    def _other(player_id: int) -> int:
        return 3 - player_id

    @staticmethod
    def _new_root(player_just_moved: int) -> MCTSNode:
        return MCTSNode(move=None, parent=None, player_just_moved=player_just_moved)

    def _commit_my_move(self, move: Tuple[int, int], size: int) -> Tuple[int, int]:
        self._apply_my_move_to_tracked_sets(move)
        self._last_size = size
        return move

    def _ensure_untried_moves(self, node: MCTSNode, size: int, free_cells: Set[Tuple[int, int]]) -> None:
        if node.untried_moves is None:
            node.untried_moves = self._ordered_moves(size, free_cells)

    def reset(self) -> None:
        self.root = self.root_player = self.player = self.opp = None
        self._last_my_cells = self._last_opp_cells = self._last_free_cells = None
        self._last_size = None
        self._dsu_cache.clear()

    def _init_tracked_sets_from_board(self, board: List[List[int]], my_player: int) -> None:
        opp_player = self._other(my_player)
        size = len(board)

        my_cells: Set[Tuple[int, int]] = set()
        opp_cells: Set[Tuple[int, int]] = set()
        free_cells: Set[Tuple[int, int]] = set()

        for r in range(size):
            for c in range(size):
                cell = board[r][c]
                if cell == my_player:
                    my_cells.add((r, c))
                elif cell == opp_player:
                    opp_cells.add((r, c))
                else:
                    free_cells.add((r, c))

        self._last_my_cells = my_cells
        self._last_opp_cells = opp_cells
        self._last_free_cells = free_cells
        self._last_size = size

    def _apply_my_move_to_tracked_sets(self, move: Tuple[int, int]) -> None:
        if None in (self._last_my_cells, self._last_opp_cells, self._last_free_cells):
            return
        self._last_my_cells.add(move)
        self._last_opp_cells.discard(move)
        self._last_free_cells.discard(move)

    def _detect_new_move(
        self,
        previous_free_cells: Set[Tuple[int, int]],
        current_board: List[List[int]],
        expected_player: int,
    ) -> Optional[Tuple[int, int]]:
        # Recorremos solo celdas que estaban vacias en el turno anterior.
        detected_move: Optional[Tuple[int, int]] = None
        for r, c in previous_free_cells:
            value = current_board[r][c]
            if value == expected_player:
                if detected_move is not None:
                    # Se detectaron mas de una jugada nueva.
                    return None
                detected_move = (r, c)
            elif value != 0:
                # Estado invalido: una celda antes libre ahora no es del jugador esperado.
                return None

        return detected_move

    def sync_after_opponent_move(self, board: List[List[int]], player_to_move: int) -> None:
        self.player = player_to_move
        self.opp = self._other(player_to_move)
        # Require previous state to be available to detect the opponent move.
        if self._last_size is None or self._last_size != len(board):
            self.reset()
            return

        if None in (self._last_free_cells, self._last_my_cells, self._last_opp_cells):
            self.reset()
            return

        opponent_move = self._detect_new_move(self._last_free_cells, board, self.opp)
        if opponent_move is None:
            self.reset()
            return

        if self.root is not None:
            child = next((c for c in self.root.children if c.move == opponent_move), None)
            self.root = child if child is not None else self._new_root(self.opp)
            if child is not None:
                self.root.parent = None
        else:
            self.root = self._new_root(self.opp)
        self.root_player = player_to_move

        # Update tracked sets and free-cells for future detection.
        self._last_opp_cells.add(opponent_move)
        self._last_my_cells.discard(opponent_move)
        self._last_free_cells.discard(opponent_move)

    def _neighbors(self, size: int, r: int, c: int) -> List[Tuple[int, int]]:
        # Vecinos en un tablero hexagonal (coordenadas offset even-r).
        deltas = [(-1, -1), (-1, 0), (0, 1), (1, 0), (1, -1), (0, -1)] if r % 2 else [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (0, -1)]
        candidates = [(r + dr, c + dc) for dr, dc in deltas]
        return [(nr, nc) for nr, nc in candidates if 0 <= nr < size and 0 <= nc < size]

    def _has_connection(self, matrix: List[List[int]], player_id: int) -> bool:
        size = len(matrix)
        if size == 0 or player_id not in (1, 2):
            return False

        # Use cached DSU when available; it will be invalidated on matrix mutations.
        dsu = self._build_or_get_dsu(matrix, player_id)

        def idx(rr: int, cc: int) -> int:
            return rr * size + cc

        starts = [idx(r, 0) for r in range(size) if matrix[r][0] == player_id] if player_id == 1 else [idx(0, c) for c in range(size) if matrix[0][c] == player_id]
        ends = [idx(r, size - 1) for r in range(size) if matrix[r][size - 1] == player_id] if player_id == 1 else [idx(size - 1, c) for c in range(size) if matrix[size - 1][c] == player_id]
        if not starts or not ends:
            return False
        end_roots = {dsu.find(x) for x in ends}
        return any(dsu.find(x) in end_roots for x in starts)

    def _ordered_moves(self, size: int, free_cells: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
        # Prioriza celdas adyacentes a fichas ya colocadas y luego el resto.
        occupied: Set[Tuple[int, int]] = set()
        if self._last_my_cells is not None:
            occupied |= self._last_my_cells
        if self._last_opp_cells is not None:
            occupied |= self._last_opp_cells

        adjacent_free: Set[Tuple[int, int]] = set()
        for r, c in occupied:
            for n in self._neighbors(size, r, c):
                if n in free_cells:
                    adjacent_free.add(n)
        return list(adjacent_free) + list(free_cells - adjacent_free)

    def _mark_at(
        self,
        matrix: List[List[int]],
        free_cells: Set[Tuple[int, int]],
        r: int,
        c: int,
        player_id: Optional[int],
    ) -> None:
        # Any mutation of the matrix invalidates cached DSU for that matrix id.
        matrix_id = id(matrix)
        self._dsu_cache.pop((matrix_id, 1), None)
        self._dsu_cache.pop((matrix_id, 2), None)

        if player_id is None:
            matrix[r][c] = 0
            free_cells.add((r, c))
            return

        matrix[r][c] = player_id
        free_cells.discard((r, c))

    def _build_or_get_dsu(self, matrix: List[List[int]], player_id: int) -> "DisjointSet":
        """Return a cached DisjointSet for the given matrix and player if available,
        otherwise build and cache a new one."""
        matrix_id = id(matrix)
        key = (matrix_id, player_id)
        size = len(matrix)
        cached = self._dsu_cache.get(key)
        if cached is not None and cached[1] == size:
            return cached[0]

        # Build a new DSU for this matrix/player
        dsu = DisjointSet(size * size)

        def idx(rr: int, cc: int) -> int:
            return rr * size + cc

        for r in range(size):
            for c in range(size):
                if matrix[r][c] != player_id:
                    continue
                for nr, nc in self._neighbors(size, r, c):
                    if matrix[nr][nc] == player_id:
                        dsu.union(idx(r, c), idx(nr, nc))

        self._dsu_cache[key] = (dsu, size)
        return dsu

    def _rollout(
        self,
        matrix: List[List[int]],
        size: int,
        free_cells: Set[Tuple[int, int]],
        player_to_move: int,
        played_moves: List[Tuple[int, int]],
    ) -> int:
        """
        Simulación aleatoria hasta estado terminal.
        Retorna el ganador (1 o 2).
        """
        available = self._ordered_moves(size, free_cells)
        random.shuffle(available)

        current = player_to_move
        for r, c in available:
            self._mark_at(matrix, free_cells, r, c, current)
            played_moves.append((r, c))

            if self._has_connection(matrix, current):
                return current

            current = self._other(current)

        # En Hex no hay empates. Fallback defensivo por consistencia.
        return self._other(player_to_move)

    def best_move(self, board: HexBoard, root_player: int) -> Optional[Tuple[int, int]]:
        if self.player is not None and self.player != root_player:
            self.reset()

        self.player = root_player
        self.opp = self._other(root_player)

        size = getattr(board, "size", None)
        matrix = getattr(board, "board", None)
        if not isinstance(size, int) or size <= 0 or not isinstance(matrix, list):
            raise ValueError("MCTS.best_move espera un tablero con atributos 'size' y 'board'.")

        if (
            self._last_size != size
            or self._last_my_cells is None
            or self._last_opp_cells is None
            or self._last_free_cells is None
        ):
            # Inicializacion unica (o reinicializacion tras desincronizacion).
            self._init_tracked_sets_from_board(matrix, root_player)

        # Simulación sobre copia local para no mutar el estado real de la partida.
        sim_matrix = [row[:] for row in matrix]
        free_cells: Set[Tuple[int, int]] = set(self._last_free_cells)

        if not free_cells:
            self._last_size = size
            return None

        if self.root is None or self.root_player != root_player:
            self.root = self._new_root(self.opp)
            self.root_player = root_player

        root = self.root
        # Run iterations until time limit is reached (replace fixed iteration count).
        time_limit = 3.75
        start_time = time.perf_counter()

        while time.perf_counter() - start_time < time_limit:
            node = root
            current_player = root_player
            played_moves: List[Tuple[int, int]] = []
            winner: Optional[int] = None

            # 1) Selection
            self._ensure_untried_moves(node, size, free_cells)
            while node.untried_moves == [] and node.children:
                node = node.uct_select_child(MCTS.exploration)
                r, c = node.move
                self._mark_at(sim_matrix, free_cells, r, c, current_player)
                played_moves.append((r, c))

                # Si este movimiento ya ganó, no necesitamos expandir/simular.
                if self._has_connection(sim_matrix, current_player):
                    winner = current_player
                    break

                current_player = self._other(current_player)
                self._ensure_untried_moves(node, size, free_cells)

            # 2) Expansion
            if winner is None:
                self._ensure_untried_moves(node, size, free_cells)
                if node.untried_moves:
                    move = random.choice(node.untried_moves)
                    self._mark_at(sim_matrix, free_cells, move[0], move[1], current_player)
                    played_moves.append(move)
                    node = node.add_child(move, current_player)

                    if self._has_connection(sim_matrix, current_player):
                        winner = current_player
                    else:
                        current_player = self._other(current_player)

            # 3) Simulation
            if winner is None:
                winner = self._rollout(sim_matrix, size, free_cells, current_player, played_moves)

            # 4) Backpropagation
            while node is not None:
                node.update(winner)
                node = node.parent

            # Undo del estado aplicado durante selección/expansión/simulación.
            while played_moves:
                r, c = played_moves.pop()
                self._mark_at(sim_matrix, free_cells, r, c, None)

        if not root.children:
            move = random.choice(tuple(free_cells))
            return self._commit_my_move(move, size)

        # Mejor movimiento final: mayor número de visitas (criterio robusto).
        best_child = max(root.children, key=lambda child: child.visits)
        self.root = best_child
        self.root.parent = None
        self.root_player = self.opp
        return self._commit_my_move(best_child.move, size)