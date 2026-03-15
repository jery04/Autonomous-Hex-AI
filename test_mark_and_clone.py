import unittest

from solution import HexNodeGraph
from board import HexBoard


class TestMarkAndClone(unittest.TestCase):
    def test_mark_node_at_success_and_errors(self):
        g = HexNodeGraph()
        # si la matriz no existe debe lanzar IndexError
        with self.assertRaises(IndexError):
            g.mark_node_at(0, 0)

        # crear matriz y marcar una posición válida
        g.create_node_matrix(2, orientation=1)
        g.mark_node_at(0, 1)
        self.assertTrue(g.matrix[0][1].marked)

        # eliminar la posición y comprobar que marcar ahora lanza ValueError
        g.remove_node_at(0, 1, verbose=False)
        with self.assertRaises(ValueError):
            g.mark_node_at(0, 1)

    def test_clone_deep_copy_and_independence(self):
        size = 3
        g = HexNodeGraph()
        g.create_node_matrix(size, orientation=1)

        # marcar un nodo concreto
        g.matrix[1][1].marked = True

        # asignar un HexBoard y modificarlo para comprobar clonación
        hb = HexBoard(size)
        hb.place_piece(0, 0, 1)
        g.hex_board = hb

        # eliminar una posición para asegurar que None se preserva
        g.remove_node_at(0, 2, verbose=False)

        new = g.clone()

        # estructura básica
        self.assertIsNot(new, g)
        self.assertEqual(len(new.matrix), len(g.matrix))

        for r in range(size):
            for c in range(size):
                old = g.matrix[r][c]
                newn = new.matrix[r][c]
                if old is None:
                    self.assertIsNone(newn)
                else:
                    # nodos distintos (copia profunda por posición)
                    self.assertIsNot(old, newn)
                    self.assertEqual((newn.r, newn.c), (r, c))
                    # marcado se copia
                    self.assertEqual(newn.marked, old.marked)

                    # cada vecino en el clon debe referenciar nodos del clon
                    for neigh in newn.neighbors:
                        if neigh.r >= 0 and neigh.c >= 0:
                            self.assertIs(new.matrix[neigh.r][neigh.c], neigh)

        # extremos clonados y no compartidos
        if g.extreme1 is not None:
            self.assertIsNot(new.extreme1, g.extreme1)
            for neigh in new.extreme1.neighbors:
                if 0 <= neigh.r < size and 0 <= neigh.c < size:
                    self.assertIs(new.matrix[neigh.r][neigh.c], neigh)
        if g.extreme2 is not None:
            self.assertIsNot(new.extreme2, g.extreme2)

        # hex_board clonado por valor
        if g.hex_board is not None:
            self.assertIsNot(new.hex_board, g.hex_board)
            self.assertEqual(new.hex_board.board, g.hex_board.board)

        # independencia: modificar el clon no afecta al original
        new.matrix[1][1].marked = False
        self.assertTrue(g.matrix[1][1].marked)

    def test_mark_node_at_out_of_range_raises(self):
        g = HexNodeGraph()
        g.create_node_matrix(2, orientation=1)
        with self.assertRaises(IndexError):
            g.mark_node_at(-1, 0)
        with self.assertRaises(IndexError):
            g.mark_node_at(2, 0)

    def test_orientation_2_extremes_and_clone_preserves_links(self):
        size = 4
        g = HexNodeGraph()
        g.create_node_matrix(size, orientation=2)

        # extremos deben conectarse a la fila superior e inferior
        self.assertIsNotNone(g.extreme1)
        self.assertIsNotNone(g.extreme2)
        top_neighbors = [n for n in g.extreme1.neighbors if 0 <= n.r < size]
        bottom_neighbors = [n for n in g.extreme2.neighbors if 0 <= n.r < size]
        self.assertEqual(len(top_neighbors), size)
        self.assertEqual(len(bottom_neighbors), size)

        # clonar y comprobar que las referencias de vecinos apuntan al clon
        cloned = g.clone()
        self.assertIsNot(cloned.extreme1, g.extreme1)
        for neigh in cloned.extreme1.neighbors:
            if 0 <= neigh.r < size and 0 <= neigh.c < size:
                self.assertIs(cloned.matrix[neigh.r][neigh.c], neigh)

    def test_clone_empty_matrix_preserves_properties(self):
        g = HexNodeGraph()
        g.player = 2
        g.min_distance = 7
        hb = HexBoard(2)
        hb.place_piece(1, 1, 2)
        g.hex_board = hb

        new = g.clone()
        self.assertEqual(new.player, g.player)
        self.assertEqual(new.min_distance, g.min_distance)
        # matrix vacía en ambos
        self.assertEqual(new.matrix, [])
        # hex_board clonado
        self.assertIsNot(new.hex_board, g.hex_board)
        self.assertEqual(new.hex_board.board, g.hex_board.board)

    def test_clone_and_mark_behaviour_and_exceptions(self):
        g = HexNodeGraph()
        g.create_node_matrix(2, orientation=1)

        # remover (0,1) y asegurar que clone_and_mark propagará ValueError
        g.remove_node_at(0, 1, verbose=False)
        with self.assertRaises(ValueError):
            g.clone_and_mark(0, 1)

        # ahora prueba que clone_and_mark marca sólo la copia
        g2 = HexNodeGraph()
        g2.create_node_matrix(2, orientation=1)
        g2.matrix[0][0].marked = False
        copy = g2.clone_and_mark(0, 0)
        self.assertTrue(copy.matrix[0][0].marked)
        self.assertFalse(g2.matrix[0][0].marked)

    def test_distance_min_distance_preserved_on_clone(self):
        g = HexNodeGraph()
        g.create_node_matrix(3, orientation=1)
        # for the default full graph there should be some path; compute it
        d = g.distance_between_extremes()
        self.assertIsNotNone(d)
        self.assertEqual(g.min_distance, d)

        cloned = g.clone()
        # clone should keep min_distance value
        self.assertEqual(cloned.min_distance, g.min_distance)


if __name__ == "__main__":
    unittest.main()
