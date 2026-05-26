from __future__ import annotations
import os
import re
from pathlib import Path
import sys

# Desactivar multithreading en NumPy para matrices pequeñas
# Esto evita que Windows gaste el 99% del tiempo creando hilos innecesarios.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Añadir el directorio raíz del proyecto (Method2_Dynamic_Programming_Reformulation) al PYTHONPATH
# Esto soluciona el "ModuleNotFoundError: No module named 'src'" cuando se ejecuta manualmente
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from src.controllers.manager import Manager
from src.controllers.strategies.geometric import GeometricSIA
from src.controllers.strategies.q_nodes import QNodes
# Optional import: this project often runs only geometric strategy.
try:
    from src.controllers.strategies.phi import Phi
except Exception:
    Phi = None
import multiprocessing
import numpy as np
import pandas as pd

# Forzar utf-8 para salida en consola de Windows y evitar errores de charmap (ej. con el caracter ∅)
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')



METHOD2_ROOT = Path(__file__).resolve().parents[1]
GEOMIP_ROOT = Path(__file__).resolve().parents[3]

def convertir_a_binario(texto, n_bits=20):
    posiciones = "ABCDEFGHIJKLMNOPQRST"[:n_bits]
    binario = ["0"] * n_bits
    for letra in texto:
        if letra in posiciones:
            binario[posiciones.index(letra)] = "1"
    return "".join(binario)

def ejecutar_con_tiempo(config_sistema, condiciones, alcance, mecanismo, resultado_queue, tpm, ks=None):
    try:
        if ks is None:
            ks = [2]
        analizador_fi = GeometricSIA(config_sistema)
        resultados_k = analizador_fi.aplicar_estrategia(condiciones, alcance, mecanismo, tpm, ks=ks)
        # Convertir dict[k, Solution] a datos serializables
        datos = {}
        for k, sol in resultados_k.items():
            datos[k] = {
                "particion": sol.particion,
                "perdida": str(sol.perdida).replace('.', ','),
                "tiempo": str(sol.tiempo_ejecucion).replace('.', ','),
            }
        resultado_queue.put(datos)

    except Exception as e:
        resultado_queue.put({})

def resolver_tpm_path(estado_inicio: str) -> Path:
    """Find TPM file in common project locations based on state size."""
    sample_name = f"N{len(estado_inicio)}A.csv"
    candidates = (
        METHOD2_ROOT / "src" / ".samples" / sample_name,
        METHOD2_ROOT / ".samples" / sample_name,
        GEOMIP_ROOT / "data" / "samples" / sample_name,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No se encontró la TPM '{sample_name}'. Busqué en: {', '.join(str(c) for c in candidates)}"
    )


def inferir_estado_inicial() -> str:
    """Infer an initial state from available datasets (prefers largest NxA.csv)."""
    sample_dirs = (
        METHOD2_ROOT / "src" / ".samples",
        METHOD2_ROOT / ".samples",
        GEOMIP_ROOT / "data" / "samples",
    )
    pattern = re.compile(r"N(\d+)[A-Z]\.csv$")
    available_sizes = []

    for sample_dir in sample_dirs:
        if not sample_dir.exists():
            continue
        for sample_file in sample_dir.glob("N*.csv"):
            match = pattern.match(sample_file.name)
            if match:
                available_sizes.append(int(match.group(1)))

    if not available_sizes:
        raise FileNotFoundError("No hay archivos de muestras TPM disponibles en data/samples ni .samples.")

    n_bits = max(available_sizes)
    return "1" + ("0" * (n_bits - 1))


def ejecutar_desde_excel(
    ruta_excel: Path,
    ruta_salida: Path,
    inicio=0,
    cantidad=50,
    estado_inicio: str | None = None,
    condiciones: str | None = None,
):
    df = pd.read_excel(ruta_excel, sheet_name=8, usecols="B", skiprows=3, names=["Subsistema"]) #! here
    filas = df["Subsistema"].dropna().tolist()
    filas = filas[inicio:inicio + cantidad]
    resultados = []

    estado_inicio = estado_inicio or inferir_estado_inicial()
    condiciones = condiciones or ("1" * len(estado_inicio))
    tpm_path = resolver_tpm_path(estado_inicio)
    tpm = np.genfromtxt(tpm_path, delimiter=",")

    for i, fila in enumerate(filas, start=inicio + 1):
        partes = fila.split("|")
        if len(partes) != 2:
            continue

        alcance = convertir_a_binario(partes[0][:len(partes[0]) - 3], n_bits=len(estado_inicio))
        mecanismo = convertir_a_binario(partes[1][:len(partes[1]) - 1], n_bits=len(estado_inicio))
        print(f"Iteración {i} - Alcance: {alcance}, Mecanismo: {mecanismo}")

        config_sistema = Manager(estado_inicial=estado_inicio)

        resultado_queue = multiprocessing.Queue()
        proceso = multiprocessing.Process(target=ejecutar_con_tiempo, args=(config_sistema, condiciones, alcance, mecanismo, resultado_queue, tpm))
        
        proceso.start()
        proceso.join(timeout=3600)  

        if proceso.is_alive():
            print(f"Iteración {i} - Tiempo límite alcanzado, terminando proceso...")
            proceso.terminate()
            proceso.join()
            resultado = {"perdida": None, "tiempo": None, "particion": None}
        else:
            resultado = (
                resultado_queue.get()
                if not resultado_queue.empty()
                else {"perdida": None, "tiempo": None, "particion": None}
            )

        resultados.append({
            "Iteración": i,
            "Alcance": alcance,
            "Mecanismo": mecanismo,
            "Partición": resultado["particion"],
            "Pérdida": resultado["perdida"],
            "Tiempo de ejecución (s)": resultado["tiempo"],
        })
    df_resultados = pd.DataFrame(resultados)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    df_resultados.to_excel(ruta_salida, index=False)
    print(f"Resultados guardados en {ruta_salida}")

def probar_todos_csv(ruta_salida: Path, nombre_csv_especifico: str = None):
    """Probar los archivos CSV disponibles y guardar resultados."""
    import re

    # Obtener todos los archivos CSV únicos de ambas ubicaciones
    sample_dirs = [
        METHOD2_ROOT / "src" / ".samples",
        METHOD2_ROOT / ".samples",
        GEOMIP_ROOT / "data" / "samples",
        GEOMIP_ROOT.parent / "QNodes" / "src" / ".samples",
    ]

    csv_files = {}
    pattern = re.compile(r"N(\d+)([A-Z])\.csv$")

    for sample_dir in sample_dirs:
        if not sample_dir.exists():
            continue
        for csv_file in sample_dir.glob("N*.csv"):
            if nombre_csv_especifico and csv_file.name != nombre_csv_especifico:
                continue
            if csv_file.name not in csv_files:
                match = pattern.match(csv_file.name)
                if match:
                    csv_files[csv_file.name] = (int(match.group(1)), match.group(2), csv_file)

    # Ordenar por tamaño y letra
    csv_files_list = sorted(csv_files.values(), key=lambda x: (x[0], x[1]))

    resultados = []

    for n_bits, letra, csv_path in csv_files_list:
        print(f"\n{'='*50}")
        print(f"Probando {csv_path.name} (N={n_bits})...")
        print(f"{'='*50}")

        # Estado inicial: 1 seguido de ceros
        estado_inicial = "1" + "0" * (n_bits - 1)
        condiciones = "1" * n_bits
        alcance = "1" * n_bits
        mecanismo = "1" * n_bits

        print(f"Configuración:")
        print(f"  Estado inicial: {estado_inicial}")
        print(f"  Condiciones: {condiciones}")
        print(f"  Alcance: {alcance}")
        print(f"  Mecanismo: {mecanismo}")

        try:
            # Cargar TPM
            print(f"Cargando TPM desde {csv_path}...")
            tpm = np.genfromtxt(csv_path, delimiter=",")
            print(f"TPM cargada: {tpm.shape}")

            # Configurar sistema
            print("Configurando sistema...")
            config_sistema = Manager(estado_inicial=estado_inicial)

            # Ejecutar análisis
            print("Ejecutando análisis GeometricSIA...")
            import time
            start_time = time.time()

            max_k = min(n_bits, 5)
            ks_a_evaluar = list(range(2, max_k + 1))
            
            analizador_geo = GeometricSIA(config_sistema)
            resultados_k = analizador_geo.aplicar_estrategia(condiciones, alcance, mecanismo, tpm, ks=ks_a_evaluar)
            
            end_time = time.time()
            tiempo_real = end_time - start_time

            print("\nRESULTADO:")
            print(f"  Tiempo real medido: {tiempo_real:.4f}s")
            for k in ks_a_evaluar:
                if k in resultados_k:
                    sia_resultado = resultados_k[k]
                    print(f"  [k={k}] Pérdida: {sia_resultado.perdida:.4f} | Tiempo: {sia_resultado.tiempo_ejecucion:.4f}s")
                    print(f"         Partición: {sia_resultado.particion}")
                else:
                    print(f"  [k={k}] No disponible")

            # Guardar resultados por cada k
            for k in ks_a_evaluar:
                if k in resultados_k:
                    sia_resultado = resultados_k[k]
                    resultados.append({
                        "Muestra": csv_path.name,
                        "Tamaño": n_bits,
                        "k": k,
                        "Estado Inicial": estado_inicial,
                        "Condiciones": condiciones,
                        "Alcance": alcance,
                        "Mecanismo": mecanismo,
                        "Pérdida": sia_resultado.perdida,
                        "Tiempo (s)": sia_resultado.tiempo_ejecucion,
                        "Tiempo Real (s)": tiempo_real,
                        "Partición": sia_resultado.particion,
                    })
                else:
                    resultados.append({
                        "Muestra": csv_path.name,
                        "Tamaño": n_bits,
                        "k": k,
                        "Estado Inicial": estado_inicial,
                        "Condiciones": condiciones,
                        "Alcance": alcance,
                        "Mecanismo": mecanismo,
                        "Pérdida": None,
                        "Tiempo (s)": None,
                        "Tiempo Real (s)": None,
                        "Partición": "N/A",
                    })

            print(f"[OK] {csv_path.name} completado exitosamente")

        except Exception as e:
            print(f"\n[ERROR] en {csv_path.name}: {e}")
            import traceback
            traceback.print_exc()
            
            resultados.append({
                "Muestra": csv_path.name,
                "Tamaño": n_bits,
                "k": "N/A",
                "Estado Inicial": estado_inicial,
                "Condiciones": condiciones,
                "Alcance": alcance,
                "Mecanismo": mecanismo,
                "Pérdida": None,
                "Tiempo (s)": None,
                "Tiempo Real (s)": None,
                "Partición": f"ERROR: {str(e)}",
            })

    # Guardar resultados
    df_resultados = pd.DataFrame(resultados)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    df_resultados.to_excel(ruta_salida, index=False)
    print(f"\nResultados guardados en {ruta_salida}")
    print(f"Total de pruebas: {len(resultados)}")

    # Mostrar resumen
    print("\nResumen:")
    print("Muestra\tTamaño\tk\tTiempo (s)\tPérdida")
    print("-" * 55)
    for r in resultados:
        tiempo = f"{r['Tiempo (s)']:.4f}" if r['Tiempo (s)'] is not None else "ERROR"
        perdida = f"{r['Pérdida']:.4f}" if r['Pérdida'] is not None else "ERROR"
        print(f"{r['Muestra']}\t{r['Tamaño']}\tk={r['k']}\t{tiempo}\t{perdida}")


def iniciar():
    # Opción 1: Ejecutar desde Excel (comportamiento original)
    # ruta_entrada = Path(
    #     os.getenv(
    #         "GEOMIP_INPUT_XLSX",
    #         str(GEOMIP_ROOT / "results" / "Pruebas_Metodo2.xlsx"),
    #     )
    # )
    # ruta_salida = Path(
    #     os.getenv(
    #         "GEOMIP_OUTPUT_XLSX",
    #         str(GEOMIP_ROOT / "results" / "resultados_Geometric.xlsx"),
    #     )
    # )
    # ejecutar_desde_excel(ruta_entrada, ruta_salida)

    # Opción 2: Probar todos los CSV disponibles o uno específico
    nombre_csv = None  # Al ponerlo en None, procesará todos los CSV uno por uno
    
    if nombre_csv:
        nombre_base = nombre_csv.replace(".csv", "")
        nombre_salida = f"resultados_{nombre_base}.xlsx"
    else:
        nombre_salida = "resultados_todos_csv.xlsx"

    ruta_salida = Path(
        os.getenv(
            "GEOMIP_OUTPUT_XLSX",
            str(GEOMIP_ROOT / "results" / nombre_salida),
        )
    )
    probar_todos_csv(ruta_salida, nombre_csv_especifico=nombre_csv)

if __name__ == "__main__":
    iniciar()
