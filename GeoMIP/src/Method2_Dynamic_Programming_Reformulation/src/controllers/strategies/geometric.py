from src.models.core.system import System
from src.constants.base import NET_LABEL
from src.funcs.base import ABECEDARY
from src.middlewares.slogger import SafeLogger
from src.funcs.base import emd_efecto, seleccionar_subestado
from src.models.base.sia import SIA
from src.constants.base import (
    ACTUAL,
    EFECTO,
    TYPE_TAG,
)
from src.constants.models import (
    GEOMETRIC_ANALYSIS_TAG,
    GEOMETRIC_LABEL,
    GEOMETRIC_STRAREGY_TAG,
)
from src.controllers.manager import Manager
from src.funcs.format import fmt_k_parte_q
from src.middlewares.profile import profiler_manager, profile
from src.models.core.solution import Solution
import numpy as np
import time
from typing import List, Dict, Tuple

class GeometricSIA(SIA):
    def __init__(self, gestor: Manager):
        super().__init__(gestor)
        profiler_manager.start_session(
            f"{NET_LABEL}{len(gestor.estado_inicial)}{gestor.pagina}"
        )
        self.etiquetas = [tuple(s.lower() for s in ABECEDARY), ABECEDARY]
        self.logger = SafeLogger(GEOMETRIC_STRAREGY_TAG)
        self.tabla_transiciones: dict = {}
        self.vertices: set[tuple]
        self.memoria_particiones: dict = {}

    @profile(context={TYPE_TAG: GEOMETRIC_ANALYSIS_TAG})
    def aplicar_estrategia(
        self,
        condicion: str,
        alcance: str,
        mecanismo: str,
        tpm: np.ndarray,
        ks: list[int] = None,
    ):
        if ks is None:
            ks = [2]

        self.sia_preparar_subsistema(condicion, alcance, mecanismo, tpm)

        futuro = tuple(
            (EFECTO, efecto) for efecto in self.sia_subsistema.indices_ncubos
        )
        presente = tuple(
            (ACTUAL, actual) for actual in self.sia_subsistema.dims_ncubos
        )

        self._flat_data = np.stack([ncubo.data.ravel() for ncubo in self.sia_subsistema.ncubos])

        self.vertices = set(presente + futuro)
        dims = self.sia_subsistema.dims_ncubos
        self.estado_inicial = self.sia_subsistema.estado_inicial[dims]
        self.estado_final = 1 - self.estado_inicial

        # Limitar ks al máximo de nodos espaciales
        max_nodos = max(
            self.sia_subsistema.indices_ncubos.size,
            self.sia_subsistema.dims_ncubos.size,
        )
        ks_validos = [k for k in ks if 2 <= k <= max_nodos]

        if not ks_validos:
            return {}

        resultados = self.find_mip(ks_validos)
        return resultados
    
    def nodes_complement(self, nodes: list[tuple[int, int]]):
        return list(set(self.vertices) - set(nodes))
    
    def find_mip(self, ks: list[int]):
        """
        Implementa el algoritmo para encontrar las k-particiones óptimas
        utilizando el enfoque geométrico-topológico.
        """
        self.sia_logger.critic("empieza.")
        estado_inicial = self.estado_inicial
        estado_final = self.estado_final
        self.idx_ncubos = list(range(len(self.sia_subsistema.indices_ncubos)))
        self.caminos: Dict[int, List[List[int]]] = {0: [estado_inicial.tolist()]}
        self.tabla_transiciones[tuple(self.caminos[0][0]),tuple(self.caminos[0][0])] = [0.0 for _ in range(len(self.sia_subsistema.indices_ncubos))]
        for nivel in range(1, len(estado_inicial)+1):
            self.calcular_costos_nivel(estado_final,nivel)

        candidatos_bi = self.identificar_particiones_optimas()

        # Representación plana de vértices para bipartición recursiva (k > 2)
        vertices_all = (
            [(ACTUAL, idx) for idx in self.sia_subsistema.dims_ncubos] +
            [(EFECTO, idx) for idx in self.sia_subsistema.indices_ncubos]
        )
        N_v    = len(vertices_all)
        vtypes = np.array([v[0] for v in vertices_all], dtype=np.int8)
        vidxs  = np.array([v[1] for v in vertices_all], dtype=np.int8)

        resultados = {}

        for k in ks:
            self.memoria_particiones = {}

            if k == 2:
                # Usar candidatos originales del enfoque geométrico
                for presentes, futuros in candidatos_bi:
                    presentes_idx = self.sia_subsistema.dims_ncubos[presentes]
                    futuros_idx   = self.sia_subsistema.indices_ncubos[futuros]
                    complement_futuros_idx   = np.setdiff1d(self.sia_subsistema.indices_ncubos, futuros_idx)
                    complement_presentes_idx = np.setdiff1d(self.sia_subsistema.dims_ncubos, presentes_idx)
                    dist = self.sia_subsistema.bipartir(futuros_idx, presentes_idx).distribucion_marginal()
                    emd  = emd_efecto(dist, self.sia_dists_marginales)
                    key  = tuple([(0, n) for n in presentes_idx] + [(1, n) for n in futuros_idx])
                    P1 = [(1, n) for n in futuros_idx] + [(0, n) for n in presentes_idx]
                    P2 = [(1, n) for n in complement_futuros_idx] + [(0, n) for n in complement_presentes_idx]
                    self.memoria_particiones[key] = (emd, dist, [P1, P2])
            else:
                # Bipartición recursiva greedy con evaluación marginal incremental para k > 2.
                # En vez de llamar k_partir + distribucion_marginal por cada candidato,
                # se mantiene base_dist y solo se actualizan los NCubes afectados por el split.
                ncubo_list = self.sia_subsistema.ncubos
                estado     = self.sia_subsistema.estado_inicial
                n_ncubos   = len(ncubo_list)
                ncube_pos  = {nc.indice: i for i, nc in enumerate(ncubo_list)}

                def ncube_valor(nc, mec_set):
                    marg = nc.marginalizar(np.setdiff1d(nc.dims, list(mec_set)))
                    if marg.dims.size:
                        return 1.0 - float(marg.data[seleccionar_subestado(
                            tuple(estado[j] for j in marg.dims)
                        )])
                    return 1.0 - float(marg.data)

                def build_group_info(grps):
                    g_alc, g_mec, nc_grp = [], [], {}
                    for g, gv in enumerate(grps):
                        alc, mec = [], set()
                        for pos in gv:
                            if vtypes[pos] == EFECTO:
                                alc.append(int(vidxs[pos]))
                                nc_grp[int(vidxs[pos])] = g
                            else:
                                mec.add(int(vidxs[pos]))
                        g_alc.append(alc)
                        g_mec.append(frozenset(mec))
                    return g_alc, g_mec, nc_grp

                def eval_isolate(pos, gi, base_dist, g_alc, g_mec):
                    new_dist = base_dist.copy()
                    v_time   = int(vtypes[pos])
                    v_idx    = int(vidxs[pos])
                    if v_time == EFECTO:
                        nc_i = ncube_pos[v_idx]
                        new_dist[nc_i] = ncube_valor(ncubo_list[nc_i], frozenset())
                    else:
                        new_mec = g_mec[gi] - {v_idx}
                        for nc_idx in g_alc[gi]:
                            nc_i = ncube_pos[nc_idx]
                            new_dist[nc_i] = ncube_valor(ncubo_list[nc_i], new_mec)
                    return emd_efecto(new_dist, self.sia_dists_marginales), new_dist

                groups = [list(range(N_v))]
                g_alc, g_mec, nc_grp = build_group_info(groups)
                base_dist = np.array(
                    [ncube_valor(nc, g_mec[nc_grp[nc.indice]]) for nc in ncubo_list],
                    dtype=np.float32,
                )
                best_emd_k, best_dist_k = float('inf'), None

                for _ in range(k - 1):
                    best_split, best_ng, best_new_base = float('inf'), None, None

                    for gi, group in enumerate(groups):
                        if len(group) < 2:
                            continue
                        for pos in group:
                            emd, new_dist = eval_isolate(pos, gi, base_dist, g_alc, g_mec)
                            if emd < best_split:
                                best_split    = emd
                                best_new_base = new_dist
                                ng = [g[:] for g in groups]
                                ng[gi] = [p for p in group if p != pos]
                                ng.append([pos])
                                best_ng = ng

                    groups     = best_ng
                    best_emd_k = best_split
                    best_dist_k = best_new_base
                    g_alc, g_mec, nc_grp = build_group_info(groups)
                    base_dist = best_new_base

                partes_display = [
                    [vertices_all[vv] for vv in gv]
                    for gv in groups
                ]
                key = tuple(v for gv in groups for v in [vertices_all[vv] for vv in gv])
                self.memoria_particiones[key] = (best_emd_k, best_dist_k, partes_display)

            if not self.memoria_particiones:
                continue

            mip_k = min(self.memoria_particiones, key=lambda k2: self.memoria_particiones[k2][0])
            partes_fmt = self.memoria_particiones[mip_k][2]
            fmt_k = fmt_k_parte_q(partes_fmt)

            resultados[k] = Solution(
                estrategia=f"{GEOMETRIC_LABEL} (k={k})",
                perdida=self.memoria_particiones[mip_k][0],
                distribucion_subsistema=self.sia_dists_marginales,
                distribucion_particion=self.memoria_particiones[mip_k][1],
                tiempo_total=time.time() - self.sia_tiempo_inicio,
                particion=fmt_k,
            )

        return resultados
    
    def calcular_costos_nivel(self,estado_final: np.ndarray, nivel):
        n = len(estado_final)      
        visitados:set[tuple] = set()
        self.caminos[nivel] = []
        for estado_anterior in self.caminos[nivel - 1]:
            estado_actual = np.array(estado_anterior)
            for i in range(n):
                if estado_actual[i] != estado_final[i]:
                    nuevo_estado = estado_actual.copy()
                    nuevo_estado[i] = estado_final[i]
                    nuevo_estado_tuple = tuple(nuevo_estado)
                    if nuevo_estado_tuple not in visitados:
                        self.caminos[nivel].append(nuevo_estado.tolist())
                        self.calcular_costo(self.caminos[0][0],nuevo_estado.tolist(),self.idx_ncubos)
                        visitados.add(nuevo_estado_tuple)

    def calcular_costo(self, estado_inicial:tuple, estado_final:tuple, ncubos:list[int]):
        """
            Funcion encargada de calcular el costo de transicion de transicion del estado inicial al estado final
            para las variables futuras definidas en ncubos
            aplica la funcion de costo tx(i,j)= y(|X[i]-X[j]|+ sum(tx(k,j)))
            donde:
                - y es el factor de decrecimiento 1/2^(dh(i,j))
                - dh(i,j) es la distancia hamming entre i y j
                - X[i] es el valor de probabilida de transicion de un estado para cada variable futura
                - sum(tx(i,k)) son todos costos de transicion de los vecinos de j que estan en un 
                  camino optimo desde i
        """
        key = tuple(estado_inicial), tuple(estado_final)
        distancia_hamming = self.hamming(estado_inicial, estado_final)
        factor = 1 / (2 ** distancia_hamming)

        estado_ini_int = int("".join(map(str, estado_inicial[::-1])), 2)
        estado_fin_int = int("".join(map(str, estado_final[::-1])), 2)

        diffs = np.abs(self._flat_data[:, estado_ini_int] - self._flat_data[:, estado_fin_int])
        self.tabla_transiciones[key] = diffs.tolist()

        if distancia_hamming > 1:
            for i in range(len(estado_inicial)):
                if estado_inicial[i] != estado_final[i]:
                    nuevo_estado = estado_final.copy()
                    nuevo_estado[i] = estado_inicial[i]
                    temp_key = tuple(estado_inicial), tuple(nuevo_estado)
                    for n in ncubos:
                        self.tabla_transiciones[key][n] += self.tabla_transiciones[temp_key][n]

        self.tabla_transiciones[key] = [factor * n for n in self.tabla_transiciones[key]]

    def identificar_particiones_optimas(self):
        """
        Identifica las particiones óptimas basadas en los costos de transición
        y las distancias Hamming entre los estados.
        """
        n_vars = len(self.tabla_transiciones[tuple(self.caminos[0][0]), tuple(self.estado_final)])

        candidatos = [
            [list(range(len(self.estado_final))), [i for i in range(n_vars) if i != idx]]
            for idx in range(n_vars)
        ]

        mitad = (len(self.caminos) // 2) + (len(self.caminos) % 2)
        for nivel in range(1, mitad):
            costo_candidato_nivel = np.inf
            presentes_nivel = []
            futuros_nivel = []
            for estado in self.caminos[nivel]:
                actual = self.tabla_transiciones.get((tuple(self.caminos[0][0]), tuple(estado)))
                estado_complementario = (1 - np.array(estado)).tolist()
                complementario = self.tabla_transiciones.get((tuple(self.caminos[0][0]), tuple(estado_complementario)))
                if actual is None or complementario is None:
                    continue
                costo_candidato = 0
                presentes = []
                futuros = []
                for idx, i in enumerate(estado):
                    if i == self.caminos[0][0][idx]:
                        presentes.append(idx)
                for idx in range(len(self.idx_ncubos)):
                    if actual[idx] <= complementario[idx]:
                        futuros.append(idx)
                        costo_candidato += actual[idx]
                    else:
                        costo_candidato += complementario[idx]
                if costo_candidato < costo_candidato_nivel:
                    costo_candidato_nivel = costo_candidato
                    presentes_nivel = presentes
                    futuros_nivel = futuros
            candidatos.append([presentes_nivel, futuros_nivel])
        return candidatos

    def hamming(self,a: List[int], b: List[int]) -> int:
        return sum(x != y for x, y in zip(a, b))