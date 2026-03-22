from typing import Any, List, Optional, Set, Tuple
import math
import random

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
    base_iterations = 900
    exploration = 1.41421356237

    @staticmethod
    def _other(player_id: int) -> int:
        return 2 if player_id == 1 else 1

    @staticmethod
    def _neighbors(size: int, r: int, c: int) -> List[Tuple[int, int]]:
        # Vecinos en un tablero hexagonal (coordenadas offset even-r).
        if r % 2 != 0:
            candidates = [
                (r - 1, c - 1),    # arriba-izquierda   NW
                (r - 1, c    ),    # arriba-derecha     NE
                (r    , c + 1),    # derecha            E
                (r + 1, c    ),    # abajo-derecha      SE
                (r + 1, c - 1),    # abajo-izquierda    SW
                (r    , c - 1),    # izquierda          W
            ]
        else:
            candidates = [
                (r - 1, c    ),    # arriba-izquierda   NW
                (r - 1, c + 1),    # arriba-derecha     NE
                (r    , c + 1),    # derecha            E
                (r + 1, c + 1),    # abajo-derecha      SE
                (r + 1, c    ),    # abajo-izquierda    SW
                (r    , c - 1),    # izquierda          W
            ]
        return [(nr, nc) for nr, nc in candidates if 0 <= nr < size and 0 <= nc < size]

    @staticmethod
    def _has_connection(matrix: List[List[int]], player_id: int) -> bool:
        size = len(matrix)
        if size == 0:
            return False

        visited = [[False] * size for _ in range(size)]
        stack: List[Tuple[int, int]] = []

        if player_id == 1:
            for r in range(size):
                if matrix[r][0] == player_id:
                    visited[r][0] = True
                    stack.append((r, 0))

            target_col = size - 1
            while stack:
                r, c = stack.pop()
                if c == target_col:
                    return True
                for nr, nc in MCTS._neighbors(size, r, c):
                    if not visited[nr][nc] and matrix[nr][nc] == player_id:
                        visited[nr][nc] = True
                        stack.append((nr, nc))
            return False

        if player_id == 2:
            for c in range(size):
                if matrix[0][c] == player_id:
                    visited[0][c] = True
                    stack.append((0, c))

            target_row = size - 1
            while stack:
                r, c = stack.pop()
                if r == target_row:
                    return True
                for nr, nc in MCTS._neighbors(size, r, c):
                    if not visited[nr][nc] and matrix[nr][nc] == player_id:
                        visited[nr][nc] = True
                        stack.append((nr, nc))
            return False

        return False

    @staticmethod
    def _ordered_moves(size: int, free_cells: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
        # Prioriza centro para mejorar convergencia al inicio.
        center = (size - 1) / 2.0
        return sorted(
            free_cells,
            key=lambda rc: (abs(rc[0] - center) + abs(rc[1] - center), rc[0], rc[1]),
        )

    @staticmethod
    def _mark_at(
        matrix: List[List[int]],
        free_cells: Set[Tuple[int, int]],
        r: int,
        c: int,
        player_id: Optional[int],
    ) -> None:
        if player_id is None:
            matrix[r][c] = 0
            free_cells.add((r, c))
            return

        matrix[r][c] = player_id
        free_cells.discard((r, c))

    @staticmethod
    def _init_untried_moves(
        node: MCTSNode,
        size: int,
        free_cells: Set[Tuple[int, int]],
    ) -> None:
        if node.untried_moves is None:
            node.untried_moves = MCTS._ordered_moves(size, free_cells)

    @staticmethod
    def _rollout(
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
        available = MCTS._ordered_moves(size, free_cells)
        random.shuffle(available)

        current = player_to_move
        for r, c in available:
            MCTS._mark_at(matrix, free_cells, r, c, current)
            played_moves.append((r, c))

            if MCTS._has_connection(matrix, current):
                return current

            current = MCTS._other(current)

        # En Hex no hay empates. Fallback defensivo por consistencia.
        return MCTS._other(player_to_move)

    @staticmethod
    def _iteration_count(size: int, free_cells: int) -> int:
        # Menos iteraciones en tableros muy grandes para mantener respuesta fluida.
        size_factor = max(0.45, 1.15 - 0.035 * (size - 9))
        fill_factor = 0.6 + 0.4 * (free_cells / (size * size))
        return max(250, int(MCTS.base_iterations * size_factor * fill_factor))

    @staticmethod
    def best_move(board: Any, root_player: int) -> Optional[Tuple[int, int]]:
        size = getattr(board, "size", None)
        matrix = getattr(board, "board", None)
        if not isinstance(size, int) or size <= 0 or not isinstance(matrix, list):
            raise ValueError("MCTS.best_move espera un tablero con atributos 'size' y 'board'.")

        # Simulación sobre copia local para no mutar el estado real de la partida.
        sim_matrix = [row[:] for row in matrix]
        free_cells: Set[Tuple[int, int]] = {
            (r, c)
            for r in range(size)
            for c in range(size)
            if sim_matrix[r][c] == 0
        }

        if not free_cells:
            return None

        root = MCTSNode(move=None, parent=None, player_just_moved=MCTS._other(root_player))
        iterations = MCTS._iteration_count(size, len(free_cells))

        for _ in range(iterations):
            node = root
            current_player = root_player
            played_moves: List[Tuple[int, int]] = []
            winner: Optional[int] = None

            # 1) Selection
            MCTS._init_untried_moves(node, size, free_cells)
            while node.untried_moves == [] and node.children:
                node = node.uct_select_child(MCTS.exploration)
                r, c = node.move
                MCTS._mark_at(sim_matrix, free_cells, r, c, current_player)
                played_moves.append((r, c))

                # Si este movimiento ya ganó, no necesitamos expandir/simular.
                if MCTS._has_connection(sim_matrix, current_player):
                    winner = current_player
                    break

                current_player = MCTS._other(current_player)
                MCTS._init_untried_moves(node, size, free_cells)

            # 2) Expansion
            if winner is None:
                MCTS._init_untried_moves(node, size, free_cells)
                if node.untried_moves:
                    move = random.choice(node.untried_moves)
                    MCTS._mark_at(sim_matrix, free_cells, move[0], move[1], current_player)
                    played_moves.append(move)
                    node = node.add_child(move, current_player)

                    if MCTS._has_connection(sim_matrix, current_player):
                        winner = current_player
                    else:
                        current_player = MCTS._other(current_player)

            # 3) Simulation
            if winner is None:
                winner = MCTS._rollout(sim_matrix, size, free_cells, current_player, played_moves)

            # 4) Backpropagation
            while node is not None:
                node.update(winner)
                node = node.parent

            # Undo del estado aplicado durante selección/expansión/simulación.
            while played_moves:
                r, c = played_moves.pop()
                MCTS._mark_at(sim_matrix, free_cells, r, c, None)

        if not root.children:
            return random.choice(tuple(free_cells))

        # Mejor movimiento final: mayor número de visitas (criterio robusto).
        best_child = max(root.children, key=lambda child: child.visits)
        return best_child.move