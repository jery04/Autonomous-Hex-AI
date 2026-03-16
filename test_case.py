from board import HexBoard

# Tablero 5x5, reproducimos las jugadas de la imagen.
# Asumimos: "Tu" = Jugador 1 (id=1), "IA" = Jugador 2 (id=2).

b = HexBoard(5)

moves = [
    ((0, 2), (1, 3)),
    ((1, 2), (2, 0)),
    ((2, 1), (3, 1)),
    ((2, 2), (3, 2)),
    ((2, 3), (3, 3)),
    ((3, 0), (3, 4)),
]

for p_move, ai_move in moves:
    b.place_piece(p_move[0], p_move[1], 2)
    b.place_piece(ai_move[0], ai_move[1], 1)

print("Board state (rows):")
for r in b.board:
    print(r)

print("Player 1 (LEFT->RIGHT) connected?", b.check_connection(1))
print("Player 2 (TOP->BOTTOM) connected?", b.check_connection(2))
