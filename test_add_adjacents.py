from solution import HexNodeGraph


def main():
    # Crear un grafo y seleccionar una celda source
    g = HexNodeGraph()
    size = 5
    g.create_node_matrix(size, orientation=1)

    r, c = 2, 2  # source
    source = g.matrix[r][c]
    target = g.extreme1  # usar el target por defecto

    # Mostrar estado antes (opcional)
    print("Antes de add_adjacents_to_node:")
    print(" - source:", source)
    print(" - target:", target)
    print(" - source.neighbors:", [repr(n) for n in source.neighbors])

    # Llamar al método bajo prueba
    g.add_adjacents_to_node(r, c, target=target)

    # Una vez finalizado, imprimir lo pedido por el usuario
    print('\nUna vez finalizado:')
    print(" - source usado:", source)
    print(" - target usado:", target)
    print(" - vecinos de target:", [repr(n) for n in target.neighbors])
    # Imprimir vecinos directos de source (finales)
    print(" - vecinos de source:", [repr(n) for n in source.neighbors])
    print(" - vecinos de los vecinos del source:")
    for neigh in source.neighbors:
        neigh_neighs = [repr(n) for n in getattr(neigh, 'neighbors', [])]
        print(f"   {repr(neigh)} -> {neigh_neighs}")


if __name__ == '__main__':
    main()
from solution import HexNodeGraph


def run_tests():
    g = HexNodeGraph()
    g.create_node_matrix(5, orientation=1)

    # Caso 1: añadir adyacentes del centro al extremo1 (por defecto)
    src = (2, 2)
    target = g.extreme1
    prev_len = len(target.neighbors)
    g.add_adjacents_to_node(*src, target=target)

    source_node = g.matrix[src[0]][src[1]]
    for neigh in list(source_node.neighbors):
        if neigh is None or neigh is source_node or neigh is target:
            continue
        assert neigh in target.neighbors, f"{neigh} no añadido a target"
        assert target in neigh.neighbors, f"target no añadido recíprocamente a {neigh}"
    # comprobar si target pertenece a los adyacentes de los adyacentes de source
    found = False
    for neigh in list(source_node.neighbors):
        if neigh is None or neigh is source_node:
            continue
        for nn in list(neigh.neighbors):
            if nn is target:
                found = True
                break
        if found:
            break
    neighbors_have_extreme1 = found

    # llamar de nuevo no debe crear duplicados
    g.add_adjacents_to_node(*src, target=target)
    assert len(target.neighbors) >= prev_len

    # Caso 2: añadir adyacentes a un nodo normal
    t = g.matrix[0][1]
    src2 = (0, 0)
    g.add_adjacents_to_node(*src2, target=t)
    source2 = g.matrix[src2[0]][src2[1]]
    for neigh in list(source2.neighbors):
        if neigh is None or neigh is source2 or neigh is t:
            continue
        assert neigh in t.neighbors, f"{neigh} no añadido a nodo objetivo t"
        assert t in neigh.neighbors, f"t no añadido recíprocamente a {neigh}"

    # Caso 3: intento sobre posición eliminada debe lanzar ValueError
    g.remove_node_at(4, 4)
    try:
        g.add_adjacents_to_node(4, 4)
        assert False, "Se esperaba ValueError para source eliminado"
    except ValueError:
        pass

    # Caso 4: coordenadas fuera de rango -> IndexError
    try:
        g.add_adjacents_to_node(10, 10)
        assert False, "Se esperaba IndexError para coordenadas fuera de rango"
    except IndexError:
        pass

    print("All add_adjacents_to_node tests passed")
    return neighbors_have_extreme1


if __name__ == '__main__':
    print(run_tests())
