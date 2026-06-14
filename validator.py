"""
validator.py — validación de parámetros de entrada.
Devuelve listas de ValidationError; no lanza excepciones.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class ValidationError:
    field: str
    message: str

    def __str__(self) -> str:
        return f"  {self.field}: {self.message}"


# ── primitivas ────────────────────────────────────────────────────────────────

def _check_bits(value: str, n: int, label: str) -> list[ValidationError]:
    if not value:
        return [ValidationError(label, "Campo vacío.")]
    if not all(c in "01" for c in value):
        return [ValidationError(label, "Solo se permiten caracteres '0' y '1'.")]
    if len(value) != n:
        return [ValidationError(label,
            f"Debe tener {n} bits — tiene {len(value)}.")]
    return []


# ── validaciones públicas ─────────────────────────────────────────────────────

def validate_tpm(tpm: np.ndarray) -> list[ValidationError]:
    if tpm.ndim != 2:
        return [ValidationError("TPM", "Debe ser una matriz 2D.")]
    rows, cols = tpm.shape
    errors = []
    if cols < 2:
        errors.append(ValidationError("TPM", "N debe ser ≥ 2 nodos."))
    elif rows != 2 ** cols:
        errors.append(ValidationError("TPM",
            f"Formato incorrecto: con N={cols} nodos se esperan {2**cols} filas "
            f"(una por cada estado posible), pero el archivo tiene {rows}."))
    if not np.isfinite(tpm).all():
        errors.append(ValidationError("TPM",
            "Contiene valores no numéricos o celdas vacías."))
    return errors


def validate_bit_fields(fields: dict, n: int) -> list[ValidationError]:
    """
    fields: {"estado": "...", "condicion": "...", "alcance": "...", "mecanismo": "..."}
    """
    labels = {
        "estado":    "Estado inicial",
        "condicion": "Condición",
        "alcance":   "Alcance",
        "mecanismo": "Mecanismo",
    }
    errors = []
    for key, val in fields.items():
        errors.extend(_check_bits(val, n, labels.get(key, key)))
    return errors


def validate_ks(ks: list[int]) -> list[ValidationError]:
    if not ks:
        return [ValidationError("K-particiones",
            "Selecciona al menos una k-partición.")]
    return []


def validate_n(n_str: str) -> tuple[int, list[ValidationError]]:
    try:
        n = int(n_str)
    except ValueError:
        return 0, [ValidationError("N", "Debe ser un entero positivo.")]
    if n < 2:
        return n, [ValidationError("N", "Debe ser ≥ 2.")]
    if n > 25:
        return n, [ValidationError("N",
            "N > 25 puede exceder la memoria disponible.")]
    return n, []


def validate_seed(seed_str: str) -> tuple[int, list[ValidationError]]:
    try:
        return int(seed_str), []
    except ValueError:
        return 0, [ValidationError("Semilla", "Debe ser un número entero.")]


def validate_excel_range(
    desde_str: str,
    hasta_str: str,
    total: int,
) -> tuple[int, int, list[ValidationError]]:
    errors = []
    try:
        desde = int(desde_str)
        hasta = int(hasta_str)
    except ValueError:
        errors.append(ValidationError("Rango de filas",
            "Los valores deben ser enteros."))
        return 1, total, errors
    if desde < 1:
        errors.append(ValidationError("Rango de filas",
            f"Fila inicio debe ser ≥ 1 (tiene {desde})."))
    if hasta < desde:
        errors.append(ValidationError("Rango de filas",
            f"Fila fin ({hasta}) debe ser ≥ fila inicio ({desde})."))
    if total > 0 and hasta > total:
        errors.append(ValidationError("Rango de filas",
            f"Fila fin ({hasta}) supera el total disponible ({total})."))
    return desde, hasta, errors
