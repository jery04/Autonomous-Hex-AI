Proyecto de Inteligencia Artificial: HEX

“Los que están lo suficientemente locos como para pensar que pueden cambiar el mundo son quienes lo cambian.”
— Steve Jobs

Nombre: Jery Rodríguez Fernández
Grupo: C312

Descripción general
------------------
Este repositorio implementa un jugador autónomo para HEX que combina estrategias basadas en Minimax para tableros pequeños/medianos y Monte Carlo Tree Search (MCTS) para tableros grandes. Además incluye un componente de búsqueda automática de parámetros mediante Algoritmo Genético para optimizar la heurística de evaluación.

Estructura (archivos explicados)
--------------------------------
- `solution.py`
  - Propósito: Orquestador principal del experimento/ejecución. Actúa como punto de entrada para lanzar partidas, configurar parámetros y seleccionar la estrategia adecuada según el tamaño del tablero y la fase de la partida.
  - Responsabilidades típicas:
    - Leer/suministrar configuración (tamaños de tablero, semilla, pesos) y opciones de ejecución.
    - Instanciar y coordinar el motor de juego, la estrategia activa (Minimax o MCTS) y los módulos auxiliares.
    - Ejecutar bucles de juego o torneos, registrar resultados y exportar estadísticas para entrenamiento o análisis.
    - Cargar y aplicar combinaciones de pesos provenientes del `genetic_algorithm.py` cuando corresponda.

- `strategy.pdy`
  - Propósito: Contiene las implementaciones de las estrategias de juego y las heurísticas de evaluación usadas por Minimax y por los rollouts de MCTS.
  - Componentes clave:
    - Implementación de Minimax con poda alfa–beta, con control de profundidad dependiente del tamaño del tablero y de la cercanía a nodos terminales.
    - Heurística informada compuesta por varias señales: noción territorial (control posicional), distancia mínima entre extremos (BFS), cálculo de componentes conexas (número y cardinalidad máxima), análisis de casillas amenazadas y ordenación priorizada de movimientos.
    - Generador de listas de movimientos priorizados: primero casillas adyacentes a fichas existentes, luego otras, y orden interno por distancia Manhattan a posiciones estratégicas (por ejemplo, centro o bordes relevantes según la fase del juego).
    - Parámetros de la heurística: vector de pesos (p1..p6) que combina las distintas señales mediante la fórmula
      h((i1,..,i5),(p1,..,p5)) = p1*i1 + p2*i2 + p3*i3 + p4*i4 + p5*i5
      y un parámetro adicional p6 que determina el umbral de cambio de enfoque territorial (centro → bordes) según el avance de la partida.
    - Soporte para reutilizar información intermedia (estructuras de grafos, componentes, DSU, caching) para acelerar evaluación y poda.

- `genetic_algorithm.py`
  - Propósito: Encontrar automáticamente combinaciones eficaces de pesos para la heurística mediante un Algoritmo Genético (GA).
  - Flujo de trabajo y elementos:
    - Representación: cada individuo es un vector de 6 números (p1..p6) que define la personalidad/estilo del jugador.
    - Evaluación (fitness): cada individuo juega m partidas contra un oponente (referencia o conjunto de rivales); el porcentaje de victorias define su aptitud.
    - Operadores evolutivos: selección de los mejores k, cruce (crossover) con probabilidad pc, y mutación con probabilidad pm para generar la siguiente generación.
    - Hiperparámetros controladores: {n (población), m (partidas por individuo), k (seleccionados), pc, pm, w (generaciones)}.
    - Salidas: el GA devuelve (o persiste) los mejores vectores de pesos encontrados. Un ejemplo de buena combinación hallada es (p1..p6) = (16.92, 2.81, 5.15, 16.76, 1, 43.41).
    - Detalle operativo: en la apertura se puede escoger aleatoriamente una combinación dentro de un conjunto preseleccionado para diversificar las aperturas.

Notas de diseño y optimizaciones
--------------------------------
- Minimax está pensado para ser eficiente: poda alfa–beta, orden prioritario de movimientos y caching de evaluaciones.
- MCTS usa optimizaciones específicas para Hex: detección rápida de victoria mediante Disjoint Set Union (DSU) y rollouts sesgados priorizando casillas adyacentes.
- El sistema está modularizado: la evaluación (heurística) es independiente de la búsqueda (Minimax/MCTS), lo que permite usar el `genetic_algorithm.py` para optimizar la heurística sin cambiar la lógica de búsqueda.

Sugerencias:
-------------------------
- Verificar y ajustar hiperparámetros del GA si se desea más exploración/explotación.
- Ejecutar torneos con distintas combinaciones encontradas para validar robustez.
