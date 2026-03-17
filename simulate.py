import tkinter as tk
from tkinter import messagebox
from math import cos, radians, sin, sqrt

from board import HexBoard
from solution import SmartPlayer


class HexGUI:
	def __init__(self, size: int = 5):
		self.size = size
		self.board = HexBoard(size=size)
		self.smart_player = SmartPlayer(player_id=1)
		self.human_id = 2
		self.ai_id = 1

		self.human_moves: list[tuple[int, int]] = []
		self.ai_moves: list[tuple[int, int]] = []
		self.game_over = False

		self.hex_radius = 28
		self.hex_rotation_deg = 90.0
		self.margin_x = 90
		self.margin_y = 70
		self.row_step = int(sqrt(3) * self.hex_radius)
		self.col_step = int(1.5 * self.hex_radius)

		self.root = tk.Tk()
		self.root.title("Hex interactivo")
		self.root.configure(bg="#f5f7fb")

		main = tk.Frame(self.root, bg="#f5f7fb", padx=16, pady=16)
		main.pack(fill="both", expand=True)

		left = tk.Frame(main, bg="#f5f7fb")
		left.pack(side="left", fill="both", expand=True)

		right = tk.Frame(main, bg="#ffffff", padx=14, pady=14, bd=1, relief="solid")
		right.pack(side="right", fill="y")

		self.status_label = tk.Label(
			left,
			text="Tu turno: haz click en una casilla vacia",
			font=("Segoe UI", 11, "bold"),
			bg="#f5f7fb",
			fg="#1f2937",
		)
		self.status_label.pack(anchor="w", pady=(0, 10))

		canvas_width = (
			self.margin_x * 2
			+ self.col_step * (self.size - 1)
			+ 2 * self.hex_radius
			+ self.col_step // 2
			+ 30
		)
		canvas_height = self.margin_y * 2 + self.row_step * (self.size - 1) + self.hex_radius + 20
		self.canvas = tk.Canvas(
			left,
			width=canvas_width,
			height=canvas_height,
			bg="#eef2ff",
			highlightthickness=0,
		)
		self.canvas.pack(fill="both", expand=True)

		tk.Label(
			right,
			text="Tuplas seleccionadas",
			font=("Segoe UI", 11, "bold"),
			bg="#ffffff",
			fg="#111827",
		).pack(anchor="w")

		self.moves_list = tk.Listbox(
			right,
			width=36,
			height=18,
			font=("Consolas", 10),
			bd=1,
			relief="solid",
		)
		self.moves_list.pack(pady=(8, 12), fill="y")

		legend = tk.Label(
			right,
			text="X = tu jugada\nO = algoritmo\nJugador 2 conecta arriba-abajo\nJugador 1 conecta izquierda-derecha",
			justify="left",
			font=("Segoe UI", 10),
			bg="#ffffff",
			fg="#374151",
		)
		legend.pack(anchor="w")

		restart_btn = tk.Button(
			right,
			text="Reiniciar partida",
			font=("Segoe UI", 10, "bold"),
			bg="#2563eb",
			fg="white",
			activebackground="#1d4ed8",
			activeforeground="white",
			command=self.reset_game,
		)
		restart_btn.pack(anchor="w", pady=(12, 0))

		self.cell_centers: dict[tuple[int, int], tuple[float, float]] = {}
		self.cell_polygons: dict[tuple[int, int], int] = {}
		self.canvas.tag_bind("hex_cell", "<Button-1>", self.on_canvas_click)

		self.redraw_board()

	def hex_points(self, cx: float, cy: float, radius: float) -> list[float]:
		angle = radians(self.hex_rotation_deg)
		cos_a = cos(angle)
		sin_a = sin(angle)
		points = []
		for dx, dy in [
			(radius, 0),
			(radius / 2, sqrt(3) * radius / 2),
			(-radius / 2, sqrt(3) * radius / 2),
			(-radius, 0),
			(-radius / 2, -sqrt(3) * radius / 2),
			(radius / 2, -sqrt(3) * radius / 2),
		]:
			rx = dx * cos_a - dy * sin_a
			ry = dx * sin_a + dy * cos_a
			points.extend([cx + rx, cy + ry])
		return points

	def _cell_color(self, value: int) -> str:
		if value == self.ai_id:
			return "#f59e0b"
		if value == self.human_id:
			return "#10b981"
		return "#ffffff"

	def _empty_cell_color(self, r: int, c: int) -> str:
		"""Color de apoyo visual para bordes en esquema even-r horizontal."""
		if r == 0 and c == 0:
			return "#f4f3bd"
		if r == 0:
			return "#e8eef6"
		if c == 0:
			return "#e9efe7"
		return "#ffffff"

	def _cell_center(self, r: int, c: int) -> tuple[float, float]:
		# even-r horizontal: filas pares desplazadas medio paso a la derecha.
		offset_x = (self.col_step / 2) if (r % 2 == 0) else 0.0
		cx = self.margin_x + c * self.col_step + offset_x
		cy = self.margin_y + r * self.row_step
		return cx, cy

	def redraw_board(self) -> None:
		self.canvas.delete("all")
		self.cell_centers.clear()
		self.cell_polygons.clear()

		for r in range(self.size):
			for c in range(self.size):
				cx, cy = self._cell_center(r, c)

				self.cell_centers[(r, c)] = (cx, cy)

				value = self.board.board[r][c]
				fill_color = self._cell_color(value) if value != 0 else self._empty_cell_color(r, c)
				polygon_id = self.canvas.create_polygon(
					self.hex_points(cx, cy, self.hex_radius),
					fill=fill_color,
					outline="#1f2937",
					width=2,
					tags=("hex_cell", f"cell_{r}_{c}"),
				)
				self.cell_polygons[(r, c)] = polygon_id

				symbol = ""
				if value == self.human_id:
					symbol = "X"
				elif value == self.ai_id:
					symbol = "O"

				label_text = f"{r}, {c}" if not symbol else f"{r}, {c}\n{symbol}"
				self.canvas.create_text(
					cx,
					cy,
					text=label_text,
					font=("Segoe UI", 11, "bold"),
					fill="#111827",
					tags=("hex_cell", f"cell_{r}_{c}"),
				)

	def refresh_moves_list(self) -> None:
		self.moves_list.delete(0, tk.END)
		turns = max(len(self.human_moves), len(self.ai_moves))
		for i in range(turns):
			h = self.human_moves[i] if i < len(self.human_moves) else "-"
			a = self.ai_moves[i] if i < len(self.ai_moves) else "-"
			self.moves_list.insert(tk.END, f"T{i + 1:>2}: Tu {h}   IA {a}")

	def on_canvas_click(self, event: tk.Event) -> None:
		if self.game_over:
			return

		items = self.canvas.find_overlapping(event.x, event.y, event.x, event.y)
		if not items:
			return

		cell_tag = None
		for item_id in reversed(items):
			tags = self.canvas.gettags(item_id)
			for tag in tags:
				if tag.startswith("cell_"):
					cell_tag = tag
					break
			if cell_tag is not None:
				break

		if cell_tag is None:
			return

		_, r_str, c_str = cell_tag.split("_")
		r = int(r_str)
		c = int(c_str)

		if self.board.board[r][c] != 0:
			self.status_label.config(text="Esa casilla ya esta ocupada")
			return

		self.play_human_turn(r, c)

	def play_human_turn(self, r: int, c: int) -> None:
		placed = self.board.place_piece(r, c, self.human_id)
		if not placed:
			self.status_label.config(text="Jugada invalida")
			return

		self.human_moves.append((r, c))
		self.redraw_board()
		self.refresh_moves_list()

		if self.board.check_connection(self.human_id):
			self.game_over = True
			self.status_label.config(text="Ganaste: conectaste arriba-abajo")
			messagebox.showinfo("Fin de partida", "Ganaste. Jugador 2 conecto sus extremos.")
			return

		self.status_label.config(text="Turno del algoritmo...")
		self.root.update_idletasks()
		self.root.after(120, self.play_ai_turn)

	def play_ai_turn(self) -> None:
		if self.game_over:
			return
        
		ai_r, ai_c = self.smart_player.play(self.board.clone())
		placed = self.board.place_piece(ai_r, ai_c, self.ai_id)
		if not placed:
			self.game_over = True
			self.status_label.config(text="El algoritmo intento una jugada invalida")
			messagebox.showerror(
				"Error de IA",
				f"El algoritmo intento jugar en una casilla invalida: ({ai_r}, {ai_c}).",
			)
			return

		self.ai_moves.append((ai_r, ai_c))
		self.redraw_board()
		self.refresh_moves_list()

		if self.board.check_connection(self.ai_id):	
			self.game_over = True
			self.status_label.config(text="Perdiste: la IA conecto izquierda-derecha")
			messagebox.showinfo("Fin de partida", "Perdiste. Jugador 1 conecto sus extremos.")
			return

		self.status_label.config(text="Tu turno: haz click en una casilla vacia")

	def reset_game(self) -> None:
		self.board = HexBoard(size=self.size)
		self.smart_player = SmartPlayer(player_id=1)
		self.human_moves.clear()
		self.ai_moves.clear()
		self.game_over = False
		self.status_label.config(text="Tu turno: haz click en una casilla vacia")
		self.redraw_board()
		self.refresh_moves_list()

	def run(self) -> None:
		self.root.mainloop()


def main() -> None:
	app = HexGUI(size=5)
	app.run()


if __name__ == "__main__":
	main()



