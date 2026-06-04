from concurrent.futures import ThreadPoolExecutor
import time
from typing import List, Dict, Tuple
import numpy as np
from numpy.typing import NDArray

from src.models.base.sia import SIA
from src.models.core.system import System
from src.models.core.solution import Solution
from src.controllers.manager import Manager
from src.funcs.base import emd_efecto
from src.funcs.format import fmt_k_parte_q
from src.constants.base import ACTUAL, EFECTO, TYPE_TAG
from src.constants.models import GEOMETRIC_ANALYSIS_TAG


def normalize_canonical(genes: NDArray[np.int64]) -> NDArray[np.int64]:
    """
    Normaliza el cromosoma para evitar clones redundantes (e.g., [0,0,1] y [1,1,0]).
    Asigna las etiquetas del grupo de izquierda a derecha en orden de aparición usando NumPy vectorizado.
    """
    # Encontrar la primera aparición de cada etiqueta de grupo
    _, first_indices = np.unique(genes, return_index=True)
    # Obtener las etiquetas en el orden en que aparecen por primera vez (de izquierda a derecha)
    sorted_unique_vals = genes[np.sort(first_indices)]
    
    # Mapeo inverso vectorizado usando un array de búsqueda
    lookup = np.empty(genes.max() + 1, dtype=np.int64)
    lookup[sorted_unique_vals] = np.arange(len(sorted_unique_vals))
    return lookup[genes]


def is_valid(genes: NDArray[np.int64], k: int) -> bool:
    """
    Verifica que el cromosoma contenga exactamente k grupos únicos.
    """
    return len(np.unique(genes)) == k


def repair(genes: NDArray[np.int64], k: int) -> NDArray[np.int64]:
    """
    Repara el cromosoma si algún grupo queda vacío, moviendo elementos de los grupos
    más grandes a los grupos vacíos, y finalmente normalizando de forma canónica.
    """
    genes = np.copy(genes)
    unique, counts = np.unique(genes, return_counts=True)
    unique_set = set(unique)
    missing = [g for g in range(k) if g not in unique_set]

    if not missing:
        return normalize_canonical(genes)

    for m_group in missing:
        # Volver a calcular frecuencias para asegurar que movemos del grupo más grande actual
        unique, counts = np.unique(genes, return_counts=True)
        largest_group = unique[np.argmax(counts)]
        indices = np.where(genes == largest_group)[0]
        # Mover un elemento al azar
        idx_to_move = np.random.choice(indices)
        genes[idx_to_move] = m_group

    return normalize_canonical(genes)


def group_based_crossover(p1: NDArray[np.int64], p2: NDArray[np.int64], k: int) -> NDArray[np.int64]:
    """
    Operador de Cruce Basado en Grupos (Group-based Crossover).
    Hereda grupos completos del Padre 1 y rellena las posiciones restantes con las del Padre 2.
    """
    n = len(p1)
    child = -np.ones(n, dtype=np.int64)

    # Decidir cuántos grupos heredar de P1
    if k > 2:
        num_inherit = np.random.randint(1, k)
    else:
        num_inherit = 1

    groups_to_inherit = np.random.choice(k, size=num_inherit, replace=False)

    # Copiar grupos heredados de P1
    for g in groups_to_inherit:
        mask = (p1 == g)
        child[mask] = g

    # Rellenar restantes con P2
    unassigned = (child == -1)
    child[unassigned] = p2[unassigned]

    # Devolver sin normalización canónica ni reparación inmediata para evitar redundancia
    # si se va a mutar a continuación. La reparación final se realiza después de la mutación.
    return child


def mutate(genes: NDArray[np.int64], k: int) -> NDArray[np.int64]:
    """
    Muta un gen aleatorio a otro grupo.
    """
    mutated = np.copy(genes)
    idx = np.random.randint(len(genes))
    current_group = mutated[idx]
    
    # Elegir un nuevo grupo diferente al actual
    possible_groups = [g for g in range(k) if g != current_group]
    if possible_groups:
        mutated[idx] = np.random.choice(possible_groups)

    return mutated


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
        population_size: int = 50,       # Grid search: mejor relación precisión/velocidad
        generations: int = 200,
        tournament_size: int = 3,         # Grid search: torneo=3 óptimo
        mutation_rate: float = 0.2,       # Grid search: mayor exploración evita mínimos locales
        early_stop_generations: int = 25, # Grid search: parada=25 asegura convergencia
    ) -> Dict[int, Solution]:
        """
        Método de clase estático / de interfaz principal para invocar la optimización genética.
        """
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
        population_size: int = 50,       # Grid search: mejor relación precisión/velocidad
        generations: int = 200,
        tournament_size: int = 3,         # Grid search: torneo=3 óptimo
        mutation_rate: float = 0.2,       # Grid search: mayor exploración evita mínimos locales
        early_stop_generations: int = 25, # Grid search: parada=25 asegura convergencia
    ) -> Dict[int, Solution]:
        if ks is None:
            ks = [2]

        # Preparar subsistema utilizando el método heredado de SIA
        self.sia_preparar_subsistema(condicion, alcance, mecanismo, tpm)

        # Definir los vértices del sistema (presente y futuro)
        futuro = tuple(
            (EFECTO, efecto) for efecto in self.sia_subsistema.indices_ncubos
        )
        presente = tuple(
            (ACTUAL, actual) for actual in self.sia_subsistema.dims_ncubos
        )
        vertices = list(presente) + list(futuro)
        N = len(vertices)
        
        # Array NumPy de vértices para indexación rápida
        vertices_arr = np.empty(len(vertices), dtype=object)
        for idx_v, val_v in enumerate(vertices):
            vertices_arr[idx_v] = val_v

        resultados = {}

        # Instanciar el pool de hilos una única vez para todo el proceso
        with ThreadPoolExecutor() as executor:
            # Ejecutar el algoritmo genético para cada valor de k
            for k in ks:
                if k < 2 or k > N:
                    continue

                # Cache para memoizar las evaluaciones de fitness y evitar recálculos redundantes
                eval_cache = {}

                def evaluate_fitness(individual: NDArray[np.int64]) -> Tuple[float, float, np.ndarray]:
                    # 1. Convertir el cromosoma a partes del sistema (alcance, mecanismo) mediante indexación NumPy
                    partes_sistema = []
                    for g in range(k):
                        group_mask = (individual == g)
                        group_vertices = vertices_arr[group_mask]
                        
                        alcance_g = [v[1] for v in group_vertices if v[0] == EFECTO]
                        mecanismo_g = [v[1] for v in group_vertices if v[0] == ACTUAL]
                        
                        partes_sistema.append(
                            (
                                np.array(alcance_g, dtype=np.int8),
                                np.array(mecanismo_g, dtype=np.int8),
                            )
                        )

                    # 2. Evaluar utilizando las operaciones vectorizadas y de caché del subsistema
                    sistema_particionado = self.sia_subsistema.k_partir(partes_sistema)
                    dist_particion = sistema_particionado.distribucion_marginal()
                    emd = emd_efecto(dist_particion, self.sia_dists_marginales)

                    # 3. Fitness como el inverso de la pérdida
                    fitness = 1.0 / (1.0 + emd)
                    return fitness, emd, dist_particion

                def evaluate_population(pop: List[NDArray[np.int64]]) -> List[Tuple[float, float, np.ndarray]]:
                    # Filtrar individuos que no están en el cache
                    to_eval = [ind for ind in pop if tuple(ind) not in eval_cache]
                    if to_eval:
                        results = list(executor.map(evaluate_fitness, to_eval))
                        for ind, res in zip(to_eval, results):
                            eval_cache[tuple(ind)] = res
                    return [eval_cache[tuple(ind)] for ind in pop]

                # 1. Inicialización de la Población (Asegurando validez inicial y normalización canónica)
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

                # 2. Bucle de evolución
                for gen in range(generations):
                    # Evaluar población
                    evals = evaluate_population(population)
                    fitnesses = [res[0] for res in evals]

                    # Encontrar el mejor de la generación actual
                    idx_best = np.argmax(fitnesses)
                    current_best_fit = fitnesses[idx_best]
                    current_best_emd = evals[idx_best][1]
                    current_best_dist = evals[idx_best][2]
                    current_best_ind = population[idx_best]

                    # Guardar si mejora globalmente
                    if current_best_fit > best_fitness:
                        best_fitness = current_best_fit
                        best_emd = current_best_emd
                        best_dist = current_best_dist
                        best_individual = np.copy(current_best_ind)
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1

                    # Criterio de parada temprana
                    if no_improvement_count >= early_stop_generations:
                        break

                    # Crear nueva generación
                    new_population = [np.copy(best_individual)]  # Elitismo

                    def tournament_select() -> NDArray[np.int64]:
                        candidates = np.random.choice(len(population), size=tournament_size, replace=False)
                        selected_idx = candidates[np.argmax([fitnesses[c] for c in candidates])]
                        return population[selected_idx]

                    while len(new_population) < population_size:
                        # Selección
                        p1 = tournament_select()
                        p2 = tournament_select()

                        # Crossover
                        child = group_based_crossover(p1, p2, k)

                        # Mutación
                        if np.random.random() < mutation_rate:
                            child = mutate(child, k)

                        # Reparación y Normalización Canónica Final al final de la generación del hijo
                        child = repair(child, k)
                        new_population.append(child)

                    population = new_population

                # Reconstruir las partes a partir del mejor cromosoma para el formateador
                partes_fmt = [[] for _ in range(k)]
                for idx_vertex, group_id in enumerate(best_individual):
                    partes_fmt[group_id].append(vertices[idx_vertex])

                # Formatear la partición resultante para la visualización de Solution
                fmt_k = fmt_k_parte_q(partes_fmt)

                resultados[k] = Solution(
                    estrategia=f"Algoritmo Genético GeoMIP (k={k})",
                    perdida=best_emd,
                    distribucion_subsistema=self.sia_dists_marginales,
                    distribucion_particion=best_dist,
                    tiempo_total=time.time() - self.sia_tiempo_inicio,
                    particion=fmt_k,
                )

        return resultados
