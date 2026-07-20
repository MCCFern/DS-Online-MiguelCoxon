> **Nota de actualización (2026-07-12):** este documento es el resumen ejecutivo
> preparado para la planning session del equipo (repo `ML_Spa_M003`). Desde que se
> escribió, el equipo ya avanzó sobre el terreno y algunos puntos han quedado
> resueltos o decididos de otra forma:
>
> - **RGPD (tarea 1):** resuelto. El `.gitignore` de `ML_Spa_M003` excluye
>   `*_NOUP*`; el CSV crudo con datos personales se queda fuera del repo y solo
>   se versiona el agregado sin PII (`data/processed/*.csv`).
> - **Granularidad (pregunta 3):** decidida como **fecha × tramo horario**
>   (mañana/tarde, corte a las 14:00) — más fina que "diaria", ya implementada
>   en `transform.ipynb`.
> - **EDA y feature engineering:** ya hechos (`eda.ipynb`, `feature_engineering.ipynb`
>   en la rama `feature/feature-eng`), con validación de festivos, fechas
>   comerciales y descarte razonado de lags.
> - **Modelado (nuestro rol, Persona C):** ya hay un primer andamiaje en
>   `feature/modeling` (baseline + esquema `TimeSeriesSplit`) y `feature/evaluation`
>   (funciones de evaluación) sobre el dataset base, a la espera de que
>   `feature/preprocessing` cierre el dataset final de features.
>
> El resto del documento se deja tal cual se presentó, como registro de la
> decisión original.

---

# Resumen ejecutivo — Viabilidad de datos (planning session)

**Proyecto:** SpaML_M003 — ML sobre la ocupación de un spa en Sevilla
**Fecha:** 2026-07-09 · **Estado:** pre-reparto de tareas
**Fuente:** exportes reales del sistema de reservas (may-2024 → jun-2026)

---

## TL;DR

Los datos **son viables** para un proyecto de ML completo. El modelo fuerte es el de
**ocupación / carga de trabajo** (regresión sobre serie diaria): la señal está verificada
y es robusta. El modelo de **recurrencia de clientes como clasificación supervisada es débil**
(solo un 8% de clientes repiten visita real; 6,6% de positivos en cohorte observable) —
se propone sustituirlo por una **segmentación RFM no supervisada** como complemento,
dando un proyecto híbrido (la guía lo permite y lo valora).

---

## 1. Datos disponibles

| Fichero | Filas | Uso |
|---|---|---|
| `Informe-personalizado-ventas--2024-04-29--2026-06-30.csv` | 9.082 | **Máster** (38 columnas: pagos + reserva + contacto) |
| `Informe-personalizado-reservas--2024-04-29--2026-04-29.csv` | 8.023 | Aporta `N.º de pax` (enlaza 88,8% por ID de reserva) |
| `Informe-personalizado-ventas--2024-04-29--2026-04-29.csv` | 8.186 | Subconjunto del máster — descartar |

- 8.106 reservas únicas · 8.958 pagos / 123 reembolsos / 185 cancelaciones
- Identidad de cliente: nombre 100% · teléfono 95,3% · email 92,3% · país 99,9%
- ~6.700 clientes únicos reconstruibles (cifra estable ±4% según se use teléfono,
  email o nombre como clave → la identificación es robusta)

## 2. Cifras verificadas (correcciones al análisis previo)

| Métrica | Estimación previa | Verificado | Estado |
|---|---|---|---|
| Filas / reservas únicas | 9.082 / 8.106 | 9.082 / 8.106 | ✅ |
| Estacionalidad mensual (CV) | 0,33 | 0,33 (serie limpia) | ✅ |
| Carga sábado vs. martes | ~2× | 2,04× | ✅ |
| Reservas/día (media, mediana, máx) | 10,1 / 9 / 53 | 10,1 / 9 / 53 | ✅ |
| Clientes recurrentes | 15,2% | **8,0%** excluyendo tarjetas regalo y canceladas | ⚠️ corregido |
| Positivos para clasificador "vuelve en 6 meses" | no calculado | **6,6%** (252 de 3.794 en cohorte observable) | ⚠️ nuevo |

**Por qué la corrección:** el 15,2% contaba la compra de tarjetas regalo como "visita".
Una tarjeta regalo no es una visita del comprador; filtrándolas, la recurrencia real
de visitas cae a 8,0% (438 de 5.477 clientes).

## 3. Opciones de modelo

### Opción A — Ocupación / carga de trabajo ✅ RECOMENDADO (principal)

- **Target:** nº de reservas (o pax·minutos de servicio) por día. 788 días naturales,
  **98,5% con actividad** — serie casi continua, sin huecos.
- **Señal verificada:** estacionalidad mensual clara (picos dic-ene), patrón semanal
  fuerte (finde 2× entre semana), tendencia creciente.
- **Features:** calendario (día semana, mes, festivos), lags (t-1, t-7), medias móviles,
  mix de productos.
- **Modelado:** baseline (media móvil / DummyRegressor) → regresión lineal →
  RandomForest/GradientBoosting. **Validación temporal (`TimeSeriesSplit`)**, nunca
  KFold aleatorio. Métricas: MAE/RMSE + real-vs-predicho + residuos.
- **Historia de negocio:** planificación de personal y cabinas — nítida y defendible.
- **Cautela:** cortar la serie en la fecha de exportación (30-jun-2026); con lead time
  mediano de 1 día (40% reservan el mismo día), solo las últimas ~2 semanas están
  infra-contadas. 79 reservas con servicio posterior al export se excluyen.

### Opción B — Segmentación RFM de clientes ✅ complemento (no supervisado)

- Clustering (KMeans) sobre recencia/frecuencia/gasto por cliente (~6.700 clientes).
- No necesita target → esquiva el problema de censura y desbalanceo.
- Aporta el ángulo de fidelización al vídeo y convierte el proyecto en **híbrido**.

### Opción C — Recurrencia como clasificación ❌ descartar (o stretch goal)

- Cohorte observable: 3.794 clientes, solo **252 positivos (6,6%)** a 6 meses
  (a 12 meses: 8,5% sobre 2.532). Muestra pequeña + desbalanceo severo → métricas
  ruidosas y conclusión pobre ("casi nadie vuelve").
- Si se quisiera mantener, exigiría lógica de cohortes estricta y aun así el
  resultado sería frágil. No lo recomendamos como pieza central.

## 4. Riesgos y tareas de preparación (para el reparto)

| # | Tarea | Detalle | Lane natural |
|---|---|---|---|
| 1 | **RGPD / anonimización** ⚠️ | El `.gitignore` actual **no excluye `*.csv`** y los datos contienen nombre, teléfono y email reales de clientes. Antes de subir cualquier muestra a la repo pública: añadir `*.csv` al gitignore y generar `data_sample` anonimizada (hash de contacto, sin teléfono/email). | Todos (decisión de equipo) |
| 2 | Consolidación de ficheros | Varias líneas de pago por reserva (9.082 → 8.106); merge con pax por ID de reserva; parseo europeo (`45,00 €`, fechas `3/5/24` dayfirst) | preprocessing |
| 3 | Limpieza | Outlier pax=10.666; decidir tratamiento de reembolsos y cancelaciones | preprocessing |
| 4 | Catálogo de duraciones | Asignar minutos por producto (Masaje 30'/45' explícitos; rituales por catálogo) para medir carga en pax·minuto | feature-eng |
| 5 | Serie diaria + features calendario | Construcción del dataset de modelado del modelo principal | feature-eng / modeling |
| 6 | Tabla RFM por cliente | Base para el clustering complementario | feature-eng / modeling |

## 5. Preguntas abiertas para la planning

1. ¿Confirmamos ocupación (regresión) como modelo principal + RFM (clustering) como complemento?
2. Target del modelo principal: ¿reservas/día o pax·minutos/día? (la segunda es mejor métrica de negocio, requiere tarea 4)
3. ¿Granularidad diaria (788 puntos) o semanal (113 semanas)? Propuesta: diaria.
4. ¿Quién pide al negocio el catálogo de duraciones y el calendario de festivos/cierres?
5. ¿Cómo anonimizamos la `data_sample` pública? (propuesta: hash SHA de email/teléfono, eliminar nombre)

---
*Verificación reproducible: los scripts de chequeo están fuera de la repo (scratchpad de la sesión de análisis). Las cifras salen del fichero máster de ventas con corte al 30-jun-2026.*
