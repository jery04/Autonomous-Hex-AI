from board import HexBoard
from solution import HexNodeGraph
import sys


def main():
    size = 3
    board = HexBoard(size)
    g = HexNodeGraph()
    g.create_node_matrix(size, orientation=1)  # sets g.player

    # first call: no previous board stored -> should return None and store
    assert g.hex_board is None
    res = g.detect_opponent_move(board)
    assert res is None, f"Expected None on first call, got {res}"
    assert g.hex_board is not None

    opponent_id = 2 if g.player == 1 else 1

    # opponent places a piece
    ok = board.place_piece(1, 1, opponent_id)
    assert ok, "Failed to place opponent piece"
    res = g.detect_opponent_move(board)
    assert res == (1, 1), f"Expected (1,1), got {res}"

    # our player places a piece -> should NOT be detected as opponent move
    our_id = g.player
    ok = board.place_piece(0, 0, our_id)
    assert ok, "Failed to place our piece"
    res2 = g.detect_opponent_move(board)
    assert res2 is None, f"Expected None for our move, got {res2}"

    # another opponent move
    ok = board.place_piece(2, 2, opponent_id)
    assert ok, "Failed to place opponent piece"
    res3 = g.detect_opponent_move(board)
    assert res3 == (2, 2), f"Expected (2,2), got {res3}"

    print("OK: detect_opponent_move works")


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except AssertionError as e:
        print("TEST FAILED:", e)
        sys.exit(2)
    except Exception as e:
        print("ERROR:", e)
        sys.exit(3)
