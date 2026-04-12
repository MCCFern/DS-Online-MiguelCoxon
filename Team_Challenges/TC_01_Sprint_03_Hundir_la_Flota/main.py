from variables import DIMENSION, BARCOS
from clases import Tablero
from funciones import mensaje_bienvenida, pedir_coordenadas, turno_maquina, mostrar_tableros


def main():
    # ── Bienvenida (una sola vez) ────────────────────────────────────────
    mensaje_bienvenida()

    # ── Inicialización de tableros (una sola vez) ────────────────────────
    tablero_jugador = Tablero("Jugador", DIMENSION, BARCOS)
    tablero_maquina = Tablero("Máquina", DIMENSION, BARCOS)
    tablero_jugador.inicializar_tablero()
    tablero_maquina.inicializar_tablero()

    turno = "jugador"

    # ── Bucle principal ──────────────────────────────────────────────────
    while True:
        mostrar_tableros(tablero_jugador, tablero_maquina)

        if turno == "jugador":
            fila, col = pedir_coordenadas(DIMENSION)
            resultado = tablero_maquina.recibir_disparo(fila, col)

            if resultado == "repetido":
                print("  Ya disparaste ahí. Elige otra coordenada.")
                continue

            tablero_jugador.registrar_disparo_propio(fila, col, resultado)

            if resultado == "tocado":
                print("  ¡TOCADO! Vuelves a disparar.")
                if tablero_maquina.sin_barcos():
                    mostrar_tableros(tablero_jugador, tablero_maquina)
                    print("=" * 50)
                    print("  ¡ENHORABUENA! Has hundido toda la flota enemiga.")
                    print("=" * 50)
                    break
            else:
                print("  Agua. Turno de la máquina.")
                turno = "maquina"

        else:  # turno == "maquina"
            resultado = turno_maquina(tablero_jugador)

            if resultado == "tocado":
                print("  La máquina ha tocado uno de tus barcos. Sigue disparando.")
                if tablero_jugador.sin_barcos():
                    mostrar_tableros(tablero_jugador, tablero_maquina)
                    print("=" * 50)
                    print("  La máquina ha hundido toda tu flota. ¡Has perdido!")
                    print("=" * 50)
                    break
            else:
                print("  La máquina ha fallado. Es tu turno.")
                turno = "jugador"


if __name__ == "__main__":
    main()
