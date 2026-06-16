# KQMIP — Analizador de Integración de Información (IIT)

Implementación de tres algoritmos para calcular la k-Partición de Mínima Información Pérdida (k-MIP) sobre sistemas de Teoría de Información Integrada (IIT 3.0):

| Algoritmo | Descripción |
|---|---|
| **KQNodes** | Búsqueda exacta por fusión de nodos (cross-partitions) |
| **KGeoMIP** | Búsqueda geométrica recursiva (node-partitions, EMD Wasserstein-1) |
| **GeoMIP Genético** | Optimización genética sobre node-partitions |

## Requisitos

- Python 3.11+
- Windows (probado en Windows 11)

```bash
pip install -r requirements.txt
```

## Uso

### GUI interactiva

```bash
python main.py
```

Permite configurar TPM, estado inicial, alcance, mecanismo, algoritmo y k desde una interfaz gráfica. Los resultados se muestran en pantalla y se pueden exportar a Excel.

### Por lotes (línea de comandos)

```bash
python scripts/run_batch.py
```

Procesa múltiples pruebas desde un archivo Excel y guarda resultados en `results/`.

## Estructura del proyecto

```
main.py               # Punto de entrada — lanza la GUI
requirements.txt      # Dependencias
gui/                  # Interfaz gráfica (CustomTkinter)
  app.py              # Ventana principal
  runner.py           # Ejecución subprocess de algoritmos
  validator.py        # Validación de parámetros
  logica/             # Workers por algoritmo
scripts/
  run_batch.py        # Ejecución por lotes
results/              # Resultados generados (Excel, gráficos)
Entregables/          # Documentación final del proyecto
  manual_tecnico.pdf
  manual_de_usuario.pdf
  Documento_de_Analisis.pdf
  DatosPruebas2026_1_analisis.xlsx
docs/                 # Fuentes LaTeX y scripts de análisis
```

## Algoritmos — tiempos orientativos (alcance = sistema completo)

| Sistema | KQNodes | KGeoMIP | Genético |
|---|---|---|---|
| n = 10 | 0.03–0.15 s | <0.01–0.06 s | 0.3–2.5 s |
| n = 15 | 0.10–2.6 s | 0.01–1.4 s | 0.6–10 s |
| n = 20 | 0.3–226 s | 0.05–115 s | ~5–20 min* |
| n = 22 | 0.5–311 s | 0.08–330 s | ~30 min–2 h* |
| n = 25 | ~40 min* | ~35 min | >2 h* |

\* Estimado. KGeoMIP es el más rápido para sistemas grandes.
