from board import HexBoard

class HexBoardTester:
    def __init__(self, size: int = 5):
        self.size = size

    def print_board(self, board: HexBoard) -> None:
        for row in board.board:
            print(row)

    def test_neighbors(self) -> None:
        print("TEST: neighbors parity")
        b = HexBoard(self.size)
        # incluir esquinas, centros de borde y centro del tablero
        samples = [
            (0, 0), (0, 2), (0, 4),  # fila superior: izquierda, centro, derecha
            (2, 0), (2, 2), (2, 4),  # fila central: izquierda, centro, derecha
            (4, 0), (4, 2), (4, 4),  # fila inferior: izquierda, centro, derecha
        ]
        for r, c in samples:
            neigh = [(nr, nc) for nr, nc in b._neighbors(r, c) if 0 <= nr < b.size and 0 <= nc < b.size]
            print(f"cell {(r, c)} -> neighbors {neigh}")

    def test_connection_player1(self) -> None:
        print("\nTEST: connection player 1 (left->right)")
        b = HexBoard(self.size)
        for c in range(b.size):
            b.place_piece(2, c, 1)
        self.print_board(b)
        print("connected:", b.check_connection(1))

    def test_connection_player2(self) -> None:
        print("\nTEST: connection player 2 (top->bottom)")
        b = HexBoard(self.size)
        for r in range(b.size):
            b.place_piece(r, 2, 2)
        self.print_board(b)
        print("connected:", b.check_connection(2))

    def test_parity_path(self) -> None:
        print("\nTEST: parity-dependent path")
        b = HexBoard(self.size)
        coords = [(0, 1), (1, 1), (2, 2), (3, 2), (4, 2), (2, 1)]
        for r, c in coords:
            b.place_piece(r, c, 1)
        self.print_board(b)
        print("connected:", b.check_connection(1))

    def test_connection_snake(self) -> None:
        print("\nTEST: snake-like path player 1 (left->right)")
        b = HexBoard(self.size)
        coords = [(0, 2), (1, 2), (1, 1), (2, 1), (2, 2), (3, 2), (3, 1), (4, 1)]
        for r, c in coords:
            b.place_piece(r, c, 1)
        self.print_board(b)
        print("connected:", b.check_connection(1))

    def test_connection_blocked_player1(self) -> None:
        print("\nTEST: blocked path for player 1 (left->right)")
        b = HexBoard(self.size)
        for c in range(b.size):
            if c != 2:
                b.place_piece(2, c, 1)
        b.place_piece(2, 2, 2)  # opponent blocks the central cell
        self.print_board(b)
        print("connected:", b.check_connection(1))

    def test_connection_blocked_player2(self) -> None:
        print("\nTEST: blocked path for player 2 (top->bottom)")
        b = HexBoard(self.size)
        for r in range(b.size):
            if r != 2:
                b.place_piece(r, 2, 2)
        b.place_piece(2, 2, 1)  # opponent blocks the central cell
        self.print_board(b)
        print("connected:", b.check_connection(2))


if __name__ == "__main__":
    tester = HexBoardTester(size=5)
    tester.test_neighbors()
    tester.test_connection_player1()
    tester.test_connection_player2()
    tester.test_parity_path()
    tester.test_connection_snake()
    tester.test_connection_blocked_player1()
    tester.test_connection_blocked_player2()
