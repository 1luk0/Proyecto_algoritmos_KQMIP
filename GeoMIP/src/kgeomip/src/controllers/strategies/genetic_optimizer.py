import time
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from numpy.typing import NDArray

from src.models.base.sia import SIA
from src.models.core.solution import Solution
from src.controllers.manager import Manager
from src.funcs.base import emd_efecto
from src.funcs.format import fmt_k_parte_q
from src.constants.base import ACTUAL, EFECTO


def normalize_canonical(genes: NDArray[np.int64]) -> NDArray[np.int64]:
    """
    Normaliza el cromosoma para evitar clones redundantes (e.g., [0,0,1] y [1,1,0]).
    Asigna las etiquetas del grupo de izquierda a derecha en orden de aparición.
    """
    _, first_indices = np.unique(genes, return_index=True)
    sorted_unique_vals = genes[np.sort(first_indices)]
    lookup = np.empty(genes.max() + 1, dtype=np.int64)
    lookup[sorted_unique_vals] = np.arange(len(sorted_unique_vals))
    return lookup[genes]


def repair(genes: NDArray[np.int64], k: int) -> NDArray[np.int64]:
    """
    Repara el cromosoma si algún grupo queda vacío y normaliza de forma canónica.
    """
    genes = np.copy(genes)
    unique = np.unique(genes)
    unique_set = set(unique)
    missing = [g for g in range(k) if g not in unique_set]

    if not missing:
        return normalize_canonical(genes)

    for m_group in missing:
        unique, counts = np.unique(genes, return_counts=True)
        largest_group = unique[np.argmax(counts)]
        indices = np.where(genes == largest_group)[0]
        idx_to_move = np.random.choice(indices)
        genes[idx_to_move] = m_group

    return normalize_canonical(genes)


def group_based_crossover(p1: NDArray[np.int64], p2: NDArray[np.int64], k: int) -> NDArray[np.int64]:
    """
    Operador de Cruce Basado en Grupos: hereda grupos completos de P1 y rellena con P2.
    """
    n = len(p1)
    child = -np.ones(n, dtype=np.int64)

    num_inherit = np.random.randint(1, k) if k > 2 else 1
    groups_to_inherit = np.random.choice(k, size=num_inherit, replace=False)

    for g in groups_to_inherit:
        mask = (p1 == g)
        child[mask] = g

    unassigned = (child == -1)
    child[unassigned] = p2[unassigned]
    return child


def mutate(genes: NDArray[np.int64], k: int) -> NDArray[np.int64]:
    mutated = np.copy(genes)
    idx = np.random.randint(len(genes))
    current_group = mutated[idx]
    possible_groups = [g for g in range(k) if g != current_group]
    if possible_groups:
        mutated[idx] = np.random.choice(possible_groups)
    return mutated


def _params_adaptativos(N: int) -> Tuple[int, int, int]:
    """
    Retorna (population_size, generations, early_stop) según el tamaño del sistema.
    Equilibrio entre velocidad y calidad de la solución.
    """
    if N <= 20:
        return 50, 150, 20
    elif N <= 30:
        return 70, 120, 22
    else:
        return 100, 100, 25


class GeneticKGeoMIP(SIA):
    """
    Estrategia de Optimización Genética para k-particiones en GeoMIP.
    """

    def __init__(self, gestor: Manager):
        super().__init__(gestor)
        self.logger = self.sia_logger

    @classmethod
    def optimize(
        cls,
        gestor: Manager,
        condicion: str,
        alcance: str,
        mecanismo: str,
        tpm: np.ndarray,
        ks: List[int],
        population_size: Optional[int] = None,
        generations: Optional[int] = None,
        tournament_size: int = 3,
        mutation_rate: float = 0.2,
        early_stop_generations: Optional[int] = None,
    ) -> Dict[int, Solution]:
        optimizer = cls(gestor)
        return optimizer.aplicar_estrategia(
            condicion=condicion,
            alcance=alcance,
            mecanismo=mecanismo,
            tpm=tpm,
            ks=ks,
            population_size=population_size,
            generations=generations,
            tournament_size=tournament_size,
            mutation_rate=mutation_rate,
            early_stop_generations=early_stop_generations,
        )

    def aplicar_estrategia(
        self,
        condicion: str,
        alcance: str,
        mecanismo: str,
        tpm: np.ndarray,
        ks: List[int] = None,
        population_size: Optional[int] = None,
        generations: Optional[int] = None,
        tournament_size: int = 3,
        mutation_rate: float = 0.2,
        early_stop_generations: Optional[int] = None,
    ) -> Dict[int, Solution]:
        if ks is None:
            ks = [2]

        self.sia_preparar_subsistema(condicion, alcance, mecanismo, tpm)

        futuro = tuple(
            (EFECTO, efecto) for efecto in self.sia_subsistema.indices_ncubos
        )
        presente = tuple(
            (ACTUAL, actual) for actual in self.sia_subsistema.dims_ncubos
        )
        vertices = list(presente) + list(futuro)
        N = len(vertices)

        # Parámetros adaptativos según tamaño del sistema
        pop_auto, gen_auto, stop_auto = _params_adaptativos(N)
        if population_size is None:
            population_size = pop_auto
        if generations is None:
            generations = gen_auto
        if early_stop_generations is None:
            early_stop_generations = stop_auto

        vertex_types = np.array([v[0] for v in vertices], dtype=np.int8)
        vertex_indices = np.array([v[1] for v in vertices], dtype=np.int8)

        resultados = {}

        for k in ks:
            if k < 2 or k > N:
                continue

            t_inicio_k = time.time()

            eval_cache: Dict[tuple, Tuple[float, float, np.ndarray]] = {}

            def evaluate_fitness(individual: NDArray[np.int64]) -> Tuple[float, float, np.ndarray]:
                partes_sistema = []
                for g in range(k):
                    group_mask = (individual == g)
                    partes_sistema.append(
                        (
                            vertex_indices[group_mask & (vertex_types == EFECTO)],
                            vertex_indices[group_mask & (vertex_types == ACTUAL)],
                        )
                    )
                sistema_particionado = self.sia_subsistema.k_partir(partes_sistema)
                dist_particion = sistema_particionado.distribucion_marginal()
                emd = emd_efecto(dist_particion, self.sia_dists_marginales)
                fitness = 1.0 / (1.0 + emd)
                return fitness, emd, dist_particion

            def evaluate_population(pop: List[NDArray[np.int64]], executor: ThreadPoolExecutor) -> List[Tuple[float, float, np.ndarray]]:
                keys = [tuple(ind) for ind in pop]
                uncached_pairs = [(key, ind) for key, ind in zip(keys, pop) if key not in eval_cache]

                if uncached_pairs:
                    uncached_keys = [p[0] for p in uncached_pairs]
                    uncached_inds = [p[1] for p in uncached_pairs]
                    results = list(executor.map(evaluate_fitness, uncached_inds))
                    for key, res in zip(uncached_keys, results):
                        eval_cache[key] = res

                return [eval_cache[key] for key in keys]

            # Inicialización de la población
            population = []
            for _ in range(population_size):
                ind = np.random.randint(0, k, size=N, dtype=np.int64)
                ind = repair(ind, k)
                population.append(ind)

            best_fitness = -1.0
            best_emd = np.inf
            best_dist = None
            best_individual = None
            no_improvement_count = 0

            def tournament_select() -> NDArray[np.int64]:
                candidates = np.random.choice(len(population), size=tournament_size, replace=False)
                selected_idx = candidates[np.argmax([fitnesses[c] for c in candidates])]
                return population[selected_idx]

            with ThreadPoolExecutor(max_workers=4) as executor:
                for gen in range(generations):
                    evals = evaluate_population(population, executor)
                    fitnesses = [res[0] for res in evals]

                    idx_best = np.argmax(fitnesses)
                    current_best_fit = fitnesses[idx_best]
                    current_best_emd = evals[idx_best][1]
                    current_best_dist = evals[idx_best][2]
                    current_best_ind = population[idx_best]

                    if current_best_fit > best_fitness:
                        best_fitness = current_best_fit
                        best_emd = current_best_emd
                        best_dist = current_best_dist
                        best_individual = np.copy(current_best_ind)
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1

                    # Early stop: sin mejora o EMD prácticamente cero
                    if no_improvement_count >= early_stop_generations or best_emd < 1e-4:
                        break

                    new_population = [np.copy(best_individual)]

                    while len(new_population) < population_size:
                        p1 = tournament_select()
                        p2 = tournament_select()
                        child = group_based_crossover(p1, p2, k)
                        if np.random.random() < mutation_rate:
                            child = mutate(child, k)
                        child = repair(child, k)
                        new_population.append(child)

                    population = new_population

            if best_individual is None:
                continue

            partes_fmt = [[] for _ in range(k)]
            for idx_vertex, group_id in enumerate(best_individual):
                partes_fmt[group_id].append(vertices[idx_vertex])

            fmt_k = fmt_k_parte_q(partes_fmt)

            resultados[k] = Solution(
                estrategia=f"Algoritmo Genético GeoMIP (k={k})",
                perdida=best_emd,
                distribucion_subsistema=self.sia_dists_marginales,
                distribucion_particion=best_dist,
                tiempo_total=time.time() - t_inicio_k,
                particion=fmt_k,
            )

        return resultados
