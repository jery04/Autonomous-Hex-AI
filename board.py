from __future__ import annotations
import copy
from typing import List, Tuple

class HexBoard:
	def __init__(self, size: int):
		self.size = size  # Tamaño N del tablero (NxN)
		# Matriz NxN (0=vacío, 1=Jugador1, 2=Jugador2)
		self.board: List[List[int]] = [[0 for _ in range(size)] for _ in range(size)]

	def clone(self) -> "HexBoard":
		"""Devuelve una copia del tablero actual"""
		new = HexBoard(self.size)
		new.board = [row[:] for row in self.board]
		return new

	def place_piece(self, row: int, col: int, player_id: int) -> bool:
		"""Coloca una ficha si la casilla está vacía."""
		if 0 <= row < self.size and 0 <= col < self.size and self.board[row][col] == 0:
			self.board[row][col] = player_id
			return True
		return False

	def check_connection(self, player_id: int) -> bool:
		"""Verifica si el jugador ha conectado sus dos lados.

		Convención usada:
		- Jugador 1 (id=1) conecta LEFT (columna 0) a RIGHT (columna size-1).
		- Jugador 2 (id=2) conecta TOP (fila 0) a BOTTOM (fila size-1).
		"""
		visited = [[False] * self.size for _ in range(self.size)]
		stack: List[Tuple[int, int]] = []

		if player_id == 1:
			# partir de todas las celdas de la columna izquierda ocupadas por el jugador
			for r in range(self.size):
				if self.board[r][0] == 1:
					stack.append((r, 0))
					visited[r][0] = True
			target_col = self.size - 1
			while stack:
				r, c = stack.pop()
				if c == target_col:
					return True
				for nr, nc in self._neighbors(r, c):
					if 0 <= nr < self.size and 0 <= nc < self.size and not visited[nr][nc] and self.board[nr][nc] == player_id:
						visited[nr][nc] = True
						stack.append((nr, nc))
			return False

		elif player_id == 2:
			# partir de todas las celdas de la fila superior ocupadas por el jugador
			for c in range(self.size):
				if self.board[0][c] == 2:
					stack.append((0, c))
					visited[0][c] = True
			target_row = self.size - 1
			while stack:
				r, c = stack.pop()
				if r == target_row:
					return True
				for nr, nc in self._neighbors(r, c):
					if 0 <= nr < self.size and 0 <= nc < self.size and not visited[nr][nc] and self.board[nr][nc] == player_id:
						visited[nr][nc] = True
						stack.append((nr, nc))
			return False

		else:
			return False

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
