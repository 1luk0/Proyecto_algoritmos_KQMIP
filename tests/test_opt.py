"""
Benchmark de OPT-1+OPT-2: prueba una sola celda N=10 para medir el impacto.
"""
import sys
import time
from pathlib import Path
import numpy as np

QNODES_ROOT = Path(__file__).resolve().parents[1] / "QNodes"
sys.path.insert(0, str(QNODES_ROOT))

N = 10
np.random.seed(73)
tpm = np.random.randint(2, size=(2**N, N), dtype=np.int8).astype(float)

from src.strategies.q_nodes import QNodes

estado     = "1" * N
condicion  = "1" * N
alcance    = "1" * N
mecanismo  = "1" * N
ks         = [2, 3, 4, 5]

print(f"\n{'='*60}")
print(f"  N={N}  alc=ABCDEFGHIJ  mec=ABCDEFGHIJ  ks={ks}")
print(f"{'='*60}")

q = QNodes(tpm)
t0 = time.perf_counter()
resultados = q.aplicar_estrategia(estado, condicion, alcance, mecanismo, ks=ks)
t1 = time.perf_counter()

for k, sol in resultados.items():
    print(f"  k={k}: phi={sol.perdida:.4f}")
print(f"\n  Tiempo: {t1-t0:.3f}s")
print(f"{'='*60}\n")
