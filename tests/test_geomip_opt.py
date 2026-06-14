"""
Benchmark GeoMIP Genético Optimizado — prueba una sola celda N=10 para comparar
con resultados previos de QNodes (misma TPM, misma semilla).
"""
import sys
import time
from pathlib import Path
import numpy as np

GEOMIP_ROOT = Path(__file__).resolve().parents[1] / "GeoMIP" / "src" / "Method2_Dynamic_Programming_Reformulation"
sys.path.insert(0, str(GEOMIP_ROOT))

N = 10
np.random.seed(73)
tpm = np.random.randint(2, size=(2**N, N), dtype=np.int8).astype(float)

from src.controllers.manager import Manager
from src.controllers.strategies.genetic_optimizer import GeneticKGeoMIP

estado    = "1" * N
condicion = "1" * N
alcance   = "1" * N
mecanismo = "1" * N
ks        = [2, 3, 4, 5]

gestor = Manager(estado_inicial=estado)

print(f"\n{'='*60}")
print(f"  GeoMIP Genetico Optimizado")
print(f"  N={N}  alc=ABCDEFGHIJ  mec=ABCDEFGHIJ  ks={ks}")
print(f"{'='*60}")

t0 = time.perf_counter()
resultados = GeneticKGeoMIP.optimize(
    gestor=gestor,
    condicion=condicion,
    alcance=alcance,
    mecanismo=mecanismo,
    tpm=tpm,
    ks=ks,
)
t1 = time.perf_counter()

print(f"\n  --- GeoMIP (optimizado) ---")
for k, sol in resultados.items():
    print(f"  k={k}: phi={sol.perdida:.4f}  ({sol.tiempo_ejecucion:.2f}s)")
print(f"  Tiempo total: {t1-t0:.3f}s")

print(f"\n  --- QNodes (referencia) ---")
print(f"  k=2: phi=0.6172  k=3: phi=0.6367  k=4: phi=0.8306  k=5: phi=0.8634")
print(f"  Tiempo total QNodes: ~4.8s")

print(f"\n  --- Delta (|GeoMIP - QNodes|) ---")
referencia = {2: 0.6172, 3: 0.6367, 4: 0.8306, 5: 0.8634}
for k, sol in resultados.items():
    delta = abs(sol.perdida - referencia[k])
    pct = delta / referencia[k] * 100 if referencia[k] > 0 else 0
    print(f"  k={k}: delta={delta:.4f}  ({pct:.1f}%)")
print(f"{'='*60}\n")
