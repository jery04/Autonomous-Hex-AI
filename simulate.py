from board import HexBoard
from solution import SmartPlayer


def print_board(board: HexBoard) -> None:
	print("\nTablero actual:")
	for r in range(board.size):
		indent = " " * r
		row = []
		for c in range(board.size):
			value = board.board[r][c]
			row.append("." if value == 0 else str(value))
		print(f"{indent}{' '.join(row)}")
	print()

def ask_user_move(board: HexBoard) -> tuple[int, int] | None:
	while True:
		raw = input("Tu jugada (fila,columna) o 'q' para salir: ").strip().lower()
		if raw in {"q", "quit", "exit"}:
			return None

		raw = raw.replace(" ", "")
		parts = raw.split(",")

		if len(parts) != 2:
			print("Formato invalido. Usa: fila,columna (ejemplo: 2,3)")
			continue

		if not (parts[0].isdigit() and parts[1].isdigit()):
			print("Debes introducir numeros enteros no negativos.")
			continue

		r, c = int(parts[0]), int(parts[1])
		if not (0 <= r < board.size and 0 <= c < board.size):
			print(f"Fuera de rango. Debe estar entre 0 y {board.size - 1}.")
			continue

		if board.board[r][c] != 0:
			print("Esa casilla ya esta ocupada.")
			continue

		return r, c

def main() -> None:
	board = HexBoard(size=5)
	smart_player = SmartPlayer(player_id=1)
	human_id = 2
	ai_id = 1

	print("Hex interactivo")
	print("- Tu eres el jugador 2 (conectas arriba-abajo).")
	print("- El algoritmo es el jugador 1 (conecta izquierda-derecha).")

	while True:
		print_board(board)

        # El usuario hace su jugada (HUMAN)
		move = ask_user_move(board)
		if move is None:
			print("Juego finalizado por el usuario.")
			break

		user_r, user_c = move
		board.place_piece(user_r, user_c, human_id)

		if board.check_connection(human_id):
			print_board(board)
			print("Ganaste. Jugador 2 conecto sus extremos.")
			break

        # El algoritmo hace su jugada (IA)
		ai_r, ai_c = smart_player.play(board)
		placed = board.place_piece(ai_r, ai_c, ai_id)
		if not placed:
			print(f"El algoritmo intento jugar en una casilla invalida: ({ai_r}, {ai_c}).")
			break 

		print(f"SmartPlayer jugo en: ({ai_r}, {ai_c})")

		if board.check_connection(ai_id):
			print_board(board)
			print("Perdiste. Jugador 1 conecto sus extremos.")
			break


if __name__ == "__main__":
	main()



