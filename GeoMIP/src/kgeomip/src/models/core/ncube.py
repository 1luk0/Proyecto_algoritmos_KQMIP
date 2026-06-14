from dataclasses import dataclass, field
from numpy.typing import NDArray
import numpy as np


@dataclass(frozen=True)
class NCube:
    """
    N-cubo hace referencia a un cubo n-dimensional, donde estarán indexados según la posición de precedencia de los datos, permitiendo el rápido acceso y operación en memoria.
    - `indice`: índice original del n-cubo asociado con un literal (0:A, 1:B, 2:C, ...) que permita representabilidad en su alcance o tiempo futuro.
    - `dims`: dimensiones activas actuales del n-cubo, es aquí donde se conoce la dimensionalidad según su cantidad de elementos, de forma tal que si este en el tiempo es condicionado o marginalizado tendrá una dimensionalidad menor o igual a la original a pesar que haya una alta dimensión específica.
    - `data`: arreglo numpy con los datos indexados según la notación de origen, de ser necesario se aplica una transformación sobre estos que los reindexe si se desea otra notación particular.
    - `memo`: caché de marginalizaciones ya calculadas, evita recomputos costosos entre evaluaciones del algoritmo genético.
    """

    indice: int
    dims: NDArray[np.int8]
    data: np.ndarray
    memo: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validación de tamaño y dimensionalidad tras inicialización."""
        if self.dims.size and self.data.shape != (2,) * self.dims.size:
            raise ValueError(
                f"Forma inválida {self.data.shape} para dimensiones {self.dims}"
            )

    def condicionar(
        self,
        indices_condicionados: NDArray[np.int8],
        estado_inicial: NDArray[np.int8],
    ) -> "NCube":
        numero_dims = self.dims.size
        seleccion = [slice(None)] * numero_dims
        for condicion in indices_condicionados:
            level_arr = numero_dims - (condicion + 1)
            seleccion[level_arr] = estado_inicial[condicion]

        nuevas_dims = np.array(
            [dim for dim in self.dims if dim not in indices_condicionados],
            dtype=np.int8,
        )
        return NCube(
            data=self.data[tuple(seleccion)],
            dims=nuevas_dims,
            indice=self.indice,
        )

    def marginalizar(self, ejes: NDArray[np.int8]) -> "NCube":
        """
        Marginaliza el n-cubo en los ejes indicados. Utiliza memo para evitar
        recomputar la misma marginalización entre distintas evaluaciones del genético.
        """
        key = tuple(int(e) for e in ejes)

        if key in self.memo:
            return self.memo[key]

        marginable_axis = np.intersect1d(ejes, self.dims)
        if not marginable_axis.size:
            return self

        numero_dims = self.dims.size - 1
        ejes_locales = tuple(
            numero_dims - dim_idx
            for dim_idx, axis in enumerate(self.dims)
            if axis in marginable_axis
        )
        new_dims = np.array(
            [d for d in self.dims if d not in marginable_axis],
            dtype=np.int8,
        )
        result = NCube(
            data=np.mean(self.data, axis=ejes_locales, keepdims=False),
            dims=new_dims,
            indice=self.indice,
        )
        self.memo[key] = result
        return result

    def __str__(self) -> str:
        dims_str = f"dims={self.dims}"
        forma_str = f"shape={self.data.shape}"
        datos_str = str(self.data).replace("\n", "\n" + " " * 8)
        return (
            f"NCube(index={self.indice}):\n"
            f"    {dims_str}\n"
            f"    {forma_str}\n"
            f"    data=\n        {datos_str}"
        )
