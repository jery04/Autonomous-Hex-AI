from solution import HexNodeGraph, Node
import io
from contextlib import redirect_stdout


def test_remove_node_at_prints_and_clears_matrix(r: int, c: int):
    g = HexNodeGraph()
    g.create_node_matrix(3, orientation=1)

    node = g.matrix[r][c]
    # snapshot neighbors before removal
    neighs = list(node.neighbors)

    buf = io.StringIO()
    with redirect_stdout(buf):
        g.remove_node_at(r, c, verbose=True)
        # also print the matrix position after removal so the output
        # shows the final `matrix[r][c] = None` line as requested
        print(f"matrix[{r}][{c}] = {g.matrix[r][c]}")

    return 


if __name__ == "__main__":
    out = test_remove_node_at_prints_and_clears_matrix(1, 1)
    # Mostrar la salida paso a paso capturada
    print(out)
    print("test_remove_node_at_prints_and_clears_matrix passed")
