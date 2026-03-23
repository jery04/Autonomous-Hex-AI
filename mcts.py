from typing import List, Optional, Set, Tuple
import typing
import math
import random
import time
from board import HexBoard


# Compatibility shim: some external code performs `Optional[T] = ...`.
# Make that invalid subscription-assignment a harmless no-op.
if not hasattr(typing._SpecialForm, "__setitem__"):
    def _noop_setitem(self, key, value):
        return None
    typing._SpecialForm.__setitem__ = _noop_setitem

class DisjointSet:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: int, b: int) -> None:
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
            return (child.wins / child.visits) + exploration * math.sqrt(log_parent / child.visits)

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
    exploration = 1.41421356237

    def __init__(self, size: int, player_id: int) -> None:
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
        return 3 - player_id

    @staticmethod
    def _new_root(player_just_moved: int) -> MCTSNode:
        return MCTSNode(move=None, parent=None, player_just_moved=player_just_moved)

    def _sync_state_from_board(self, board: HexBoard) -> None:
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
        self._last_my_cells.add(move)
        #self._last_opp_cells.discard(move)
        self._last_free_cells.discard(move)
        return move

    def is_different_board(self, board: HexBoard, sample_size: int = 3) -> bool:
        if board.size != self._last_size:
            return True

        candidates = list(self._last_my_cells | self._last_opp_cells)
        if not candidates:
            return False

        sampled = random.sample(candidates, min(sample_size, len(candidates)))
        return any(board.board[r][c] == 0 for (r, c) in sampled)
    
    def _neighbors(self, r: int, c: int) -> List[Tuple[int, int]]:
        size = self._last_size
        deltas = [(-1, -1), (-1, 0), (0, 1), (1, 0), (1, -1), (0, -1)] if r % 2 != 0 else [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (0, -1)]
        candidates = [(r + dr, c + dc) for dr, dc in deltas]
        return [(nr, nc) for nr, nc in candidates if 0 <= nr < size and 0 <= nc < size]

    def _ordered_moves(self) -> List[Tuple[int, int]]:
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

        self._sync_state_from_board(board)

        if not self._last_free_cells:
            return None

        # Rebuild the search tree from the current legal position to avoid stale branches.
        self.root = self._new_root(self.opp)
        self.root_player = self.player

        root = self.root
        sim_board = [row[:] for row in board.board]

        time_limit = 3.5
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
