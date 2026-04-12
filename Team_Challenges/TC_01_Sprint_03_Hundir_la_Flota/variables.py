DIMENSION = 10

BARCOS = {
    "submarino":    {"eslora": 1, "cantidad": 4},
    "destructor":   {"eslora": 2, "cantidad": 3},
    "crucero":      {"eslora": 3, "cantidad": 2},
    "portaaviones": {"eslora": 4, "cantidad": 1},
}

# Valores de celda en el tablero
AGUA      = 0   # ~ agua sin disparar
BARCO     = 1   # O barco intacto
TOCADO    = 2   # X barco tocado
AGUA_DISP = 3   # · agua ya disparada (fallo)
