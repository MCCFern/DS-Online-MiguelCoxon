import random


def mensaje_bienvenida():
    print("=" * 50)
    print("       HUNDIR LA FLOTA — Bienvenido!")
    print("=" * 50)
    print("Reglas:")
    print("  · Tablero de 10x10. Introduce coordenadas (fila col).")
    print("  · Si aciertas un barco, vuelves a disparar.")
    print("  · Si fallas, dispara la máquina.")
    print("  · Gana quien hunda todos los barcos rivales.")
    print()
    print("Leyenda:")
    print("  ~  Agua sin disparar")
    print("  O  Barco intacto")
    print("  X  Barco tocado")
    print("  ·  Agua (fallo)")
    print("=" * 50)
    print()


def pedir_coordenadas(dimension=10):
    """Solicita al usuario una coordenada válida. Devuelve (fila, col)."""
    while True:
        try:
            entrada = input("Tu disparo — introduce fila y columna (ej: 3 7): ")
            partes = entrada.strip().split()
            if len(partes) != 2:
                raise ValueError
            fila, col = int(partes[0]), int(partes[1])
            if not (0 <= fila < dimension and 0 <= col < dimension):
                print(f"  Coordenadas fuera de rango. Usa valores entre 0 y {dimension - 1}.")
                continue
            return fila, col
        except ValueError:
            print("  Entrada inválida. Escribe dos números separados por espacio.")


def turno_maquina(tablero_jugador):
    """
    La máquina elige una coordenada aleatoria no repetida en el tablero del jugador
    y dispara. Devuelve el resultado: 'tocado' o 'agua'.
    """
    from variables import AGUA, BARCO

    dimension = tablero_jugador.dimension
    while True:
        fila = random.randint(0, dimension - 1)
        col  = random.randint(0, dimension - 1)
        celda = tablero_jugador.tablero[fila][col]
        if celda in (AGUA, BARCO):
            resultado = tablero_jugador.recibir_disparo(fila, col)
            print(f"  La máquina dispara en ({fila}, {col}) → {resultado.upper()}")
            return resultado


def mostrar_tableros(tablero_jugador, tablero_maquina):
    """Imprime ambos tableros uno al lado del otro."""
    print()
    print(f"  --- Tu tablero ({tablero_jugador.id}) ---"
          + "          "
          + f"--- Disparos al rival ({tablero_maquina.id}) ---")

    encabezado = "  " + " ".join(str(c) for c in range(tablero_jugador.dimension))
    print(f"{encabezado}          {encabezado}")

    for i in range(tablero_jugador.dimension):
        from clases import SIMBOLOS
        fila_propia  = " ".join(SIMBOLOS[v] for v in tablero_jugador.tablero[i])
        fila_rival   = " ".join(SIMBOLOS[v] for v in tablero_maquina.vista_rival[i])
        print(f"{i} {fila_propia}          {i} {fila_rival}")
    print()
