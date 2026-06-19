# 💻 Predicción de Precios de Portátiles · Kaggle *Data on the Top*

🌐 [English](README_EN-EN.md) · **Español**

> Modelo de Machine Learning que estima el precio (€) de un portátil a partir de sus
> especificaciones técnicas. El núcleo del proyecto es el **feature engineering** sobre
> datos en texto crudo y una comparativa rigurosa de modelos, de un baseline lineal a
> **XGBoost**.

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-006400)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6-9ACD32)
![pandas](https://img.shields.io/badge/pandas-3.0-150458?logo=pandas&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle&logoColor=white)

---

## 🎯 El problema

Una tienda de portátiles necesita **fijar precios de forma automática y objetiva**: dado un
equipo con unas características (CPU, RAM, pantalla, almacenamiento…), ¿cuál es su precio de
mercado justo? Es el mismo motor que hay detrás de comparadores de precios, tasadores de
segunda mano o sistemas de detección de productos mal preciados.

En términos de Machine Learning es un problema de **regresión supervisada**:

| | |
|---|---|
| **Objetivo** | `Price_in_euros` (variable continua) |
| **Datos** | ~1.300 portátiles → **912** train (con precio) + **391** test (a predecir) |
| **Métrica** | **RMSE** en euros — *cuanto menor, mejor* |

$$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

> 🔗 Competición: [Kaggle · Data on the Top](https://www.kaggle.com/competitions/data-on-the-top)

---

## 🧩 El verdadero reto: *feature engineering*

La dificultad no está en el algoritmo, sino en que **la información llega como texto crudo y
desordenado**. Un modelo no aprende de `"128GB SSD + 1TB HDD"`; sí aprende de "256 GB de SSD
y 1000 GB de HDD". Además, la variable más predictiva —la **densidad de píxeles (PPI)**— ni
siquiera existe en los datos: hay que *calcularla*.

| Columna original | Viene como… | Se transforma en… |
|---|---|---|
| `Ram` | `"8GB"` | `8` |
| `Weight` | `"1.86kg"` | `1.86` |
| `ScreenResolution` | `"IPS Panel Full HD / Touchscreen 1920x1080"` | resolución, **PPI**, flags táctil / IPS |
| `Cpu` | `"Intel Core i7 6700HQ 2.6GHz"` | marca, GHz, gama (i3/i5/i7…) |
| `Memory` | `"128GB SSD + 1TB HDD"` | GB de SSD / HDD / Flash / Híbrido + total |
| `Gpu` | `"Nvidia GeForce GTX 1050 Ti"` | marca + flag de gama alta |
| `OpSys` | `"Windows 10 S"` | familia de SO simplificada |

`Product` (480 valores únicos) se descarta por cardinalidad excesiva.

---

## 🔬 Metodología

Tres notebooks **autocontenidos** que comparten *exactamente* la misma ingeniería de
variables y se diferencian solo en el modelo. Así la comparación es justa y se cuenta una
progresión clara: *de lo simple a lo potente*.

1. **Pipeline sin *data leakage***. El parseo de texto es fila-a-fila (sin `.fit`), y el
   escalado + `OneHotEncoder` van dentro de un `Pipeline` de scikit-learn ajustado **solo con
   los datos de entrenamiento**. El mismo procesado se replica sobre `test.csv` con
   `.transform()`.
2. **Validación honesta**. RMSE estimado por **validación cruzada (5-fold)** en euros, no con
   un único split.
3. **Reentrenamiento final** sobre el 100 % de `train.csv` antes de predecir.

---

## 📈 Resultados

RMSE estimado por validación cruzada (5-fold), en euros:

| Notebook | Modelo | RMSE (€) ↓ | Rol |
|---|---|:---:|---|
| `01_submission_baseline` | Regresión Lineal (+ `log1p`) | ~355 | Baseline interpretable |
| `02_submission_rf_svr` | **Random Forest** (+ SVR) | ~290 | Modelos de ensemble clásicos |
| `03_submission_xgboost` | **XGBoost** 🏆 | **~257** | Mejor modelo |

> Del baseline al XGBoost: **−98 € de error (~28 % de mejora)**.

---

## 💡 Decisiones técnicas destacadas

- **El *feature engineering* es donde se ganan los puntos**, no el algoritmo. La PPI y el
  desglose de almacenamiento por tipo son las variables más influyentes.
- **El matiz del `log`**: transformar el objetivo a escala logarítmica **ayuda** a los modelos
  lineales y al SVR (el precio está sesgado a la derecha), pero **perjudica** al *boosting*,
  cuyo objetivo ya es el error cuadrático en euros. Por eso XGBoost entrena sobre el precio
  directo. *(Es el detalle más fino del proyecto.)*
- **Sin sobre-ingeniería**: se probaron LightGBM y *blends*, pero no superan al XGBoost
  individual dentro del ruido de la validación → la solución final es **simple y defendible**.

---

## 🗂️ Estructura del proyecto

```
TC_04_Sprint_12_Kaggle/
├── data/
│   ├── train.csv               # 912 portátiles con precio
│   ├── test.csv                # 391 portátiles a predecir
│   └── sample_submission.csv   # formato de entrega esperado
├── 01_submission_baseline.ipynb   # Regresión Lineal  → submission_01_baseline.csv
├── 02_submission_rf_svr.ipynb     # Random Forest + SVR → submission_02_randomforest.csv
├── 03_submission_xgboost.ipynb    # XGBoost (mejor)     → submission_03_xgboost.csv
├── submission_01_baseline.csv
├── submission_02_randomforest.csv
├── submission_03_xgboost.csv
└── README.md
```

---

## 🚀 Cómo reproducirlo

```bash
# 1) Instalar dependencias
pip install numpy pandas scikit-learn xgboost lightgbm matplotlib seaborn jupyter

# 2) Abrir y ejecutar cualquier notebook de principio a fin.
#    Cada uno regenera su propio submission_XX_*.csv listo para Kaggle.
jupyter notebook 03_submission_xgboost.ipynb
```

> Los CSV se leen con `encoding='latin-1'` (incluyen caracteres especiales en algunos GPU).

---

## 🛠️ Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `XGBoost` · `LightGBM` · `Matplotlib` · `seaborn`

**Competencias que demuestra:** limpieza y *parsing* de datos no estructurados ·
*feature engineering* · pipelines reproducibles sin *data leakage* · validación cruzada ·
comparación y *tuning* de modelos · comunicación de resultados.

---

## ✍️ Autor

**Miguel Coxon** — Bootcamp de Data Science
<!-- Personaliza tus enlaces: -->
[GitHub](https://github.com/MCCFern) · LinkedIn · Portfolio

---

<sub>Proyecto realizado como reto individual del Sprint 12. Dataset propiedad de la
competición de Kaggle correspondiente; se usa con fines educativos.</sub>
