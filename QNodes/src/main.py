import re
import sys

# Forzar UTF-8 en la salida estándar para compatibilidad con símbolos especiales en Windows
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

from src.controllers.manager import Manager
from src.funcs.iit import ABECEDARY
from src.constants.base import STR_ONE

# 👇 Importación de estrategias 👇 #
from src.strategies.q_nodes import QNodes
from src.models.base.application import aplicacion


def iniciar():
    """Punto de entrada"""

    # Lista de muestras: (estado_inicial, pagina)
    muestras = [
        ("11", "A"),      # N2A
        ("111", "A"),     # N3A
        ("111", "B"),     # N3B
        ("1111", "A"),    # N4A
        ("1111", "B"),    # N4B
        ("1111", "C"),    # N4C
        ("11111", "A"),   # N5A
        ("11111", "B"),   # N5B
        ("111111", "A"),  # N6A
        ("11111111", "A"),# N8A
        ("1111111111", "A"),# N10A
        ("111111111111111", "A"),# N15A
        ("111111111111111", "B"),# N15B
    ]

    resultados = []

    for estado_inicial, pagina in muestras:
        aplicacion.set_pagina_red_muestra(pagina)

        gestor_redes = Manager(estado_inicial)
        mpt = gestor_redes.cargar_red()

        analizador_q = QNodes(mpt)

        estado_uso = estado_inicial
        max_k = len(estado_uso)
        print(f"\n[{'='*40}]")
        print(f"Procesando Muestra: N{max_k}{pagina} (Nodos: {max_k})")
        print(f"[{'='*40}]")
        
        ks_a_evaluar = list(range(2, min(max_k, 5) + 1))
        
        diccionario_soluciones = analizador_q.aplicar_estrategia(
            estado_uso,
            "1" * max_k,
            "1" * max_k,
            "1" * max_k,
            ks=ks_a_evaluar
        )

        for k in ks_a_evaluar:
            if k in diccionario_soluciones:
                sol = diccionario_soluciones[k]
                particion_lineal = (
                    sol.particion
                    .replace('\n', ' / ')
                    .replace('⎛', '(')
                    .replace('⎝', '(')
                    .replace('⎞', ')')
                    .replace('⎠', ')')
                ).strip()
                particion_lineal = re.sub(r'\s+', ' ', particion_lineal)
                if particion_lineal.endswith(' /'):
                    particion_lineal = particion_lineal[:-2].strip()
                print(f" > k={k:2d} | Tiempo: {sol.tiempo_ejecucion:.4f}s | phi: {sol.perdida:.4f} | {particion_lineal}")
            else:
                print(f" > k={k:2d} | No disponible en esta evaluación.")


