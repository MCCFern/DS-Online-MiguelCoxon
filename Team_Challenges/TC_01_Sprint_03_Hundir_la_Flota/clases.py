import numpy as np
import random
from variables import AGUA, BARCO, TOCADO, AGUA_DISP

SIMBOLOS = {
    AGUA:      "~",
    BARCO:     "O",
    TOCADO:    "X",
    AGUA_DISP: "·",
}


class Tablero:
    def __init__(self, id_jugador, dimension, barcos):
        self.id = id_jugador
        self.dimension = dimension
        self.barcos = barcos

        # Tablero propio: muestra barcos + impactos recibidos
        self.tablero = np.full((dimension, dimension), AGUA, dtype=int)

        # Vista rival: solo impactos efectuados (sin ver los barcos del otro)
        self.vista_rival = np.full((dimension, dimension), AGUA, dtype=int)

        # Total de celdas con barco = suma de eslora * cantidad
        self.vidas = sum(v["eslora"] * v["cantidad"] for v in barcos.values())

    # ------------------------------------------------------------------
    # Inicialización: colocar todos los barcos aleatoriamente
    # ------------------------------------------------------------------
    def inicializar_tablero(self):
        for nombre, datos in self.barcos.items():
            eslora = datos["eslora"]
            for _ in range(datos["cantidad"]):
                colocado = False
                while not colocado:
                    fila = random.randint(0, self.dimension - 1)
                    col  = random.randint(0, self.dimension - 1)
                    orientacion = random.choice(["H", "V"])

                    celdas = self._calcular_celdas(fila, col, eslora, orientacion)
                    if celdas and self._celdas_libres(celdas):
                        for r, c in celdas:
                            self.tablero[r][c] = BARCO
                        colocado = True

    def _calcular_celdas(self, fila, col, eslora, orientacion):
        """Devuelve lista de (fila, col) que ocuparía el barco, o [] si sale del tablero."""
        celdas = []
        for i in range(eslora):
            if orientacion == "H":
                nueva_col = col + i
                if nueva_col >= self.dimension:
                    return []
                celdas.append((fila, nueva_col))
            else:  # V
                nueva_fila = fila + i
                if nueva_fila >= self.dimension:
                    return []
                celdas.append((nueva_fila, col))
        return celdas

    def _celdas_libres(self, celdas):
        """Comprueba que todas las celdas están vacías."""
        return all(self.tablero[r][c] == AGUA for r, c in celdas)

    # ------------------------------------------------------------------
    # Disparo recibido en este tablero
    # ------------------------------------------------------------------
    def recibir_disparo(self, fila, col):
        """
        Procesa un disparo en (fila, col).
        Devuelve: 'repetido', 'tocado' o 'agua'
        """
        celda = self.tablero[fila][col]

        if celda in (TOCADO, AGUA_DISP):
            return "repetido"

        if celda == BARCO:
            self.tablero[fila][col] = TOCADO
            self.vidas -= 1
            return "tocado"

        # celda == AGUA
        self.tablero[fila][col] = AGUA_DISP
        return "agua"

    # ------------------------------------------------------------------
    # Registro del disparo que este jugador efectuó al rival
    # ------------------------------------------------------------------
    def registrar_disparo_propio(self, fila, col, resultado):
        """Actualiza vista_rival con el resultado de un disparo propio."""
        if resultado == "tocado":
            self.vista_rival[fila][col] = TOCADO
        elif resultado == "agua":
            self.vista_rival[fila][col] = AGUA_DISP

    # ------------------------------------------------------------------
    # Condición de derrota
    # ------------------------------------------------------------------
    def sin_barcos(self):
        return self.vidas == 0

    # ------------------------------------------------------------------
    # Impresión
    # ------------------------------------------------------------------
    def imprimir(self, ocultar_barcos=False):
        """
        ocultar_barcos=True  → imprime vista_rival (disparos al rival sin ver sus barcos)
        ocultar_barcos=False → imprime tablero propio (barcos + impactos recibidos)
        """
        array = self.vista_rival if ocultar_barcos else self.tablero
        encabezado = "  " + " ".join(str(c) for c in range(self.dimension))
        print(encabezado)
        for i, fila in enumerate(array):
            fila_str = " ".join(SIMBOLOS[v] for v in fila)
            print(f"{i} {fila_str}")
