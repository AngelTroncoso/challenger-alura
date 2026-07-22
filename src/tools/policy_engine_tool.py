"""Motor de validación de políticas crediticias CMF e internas.

Todas las reglas, umbrales y ponderaciones se leen desde
``data/politica_credito.yaml`` para evitar valores hardcodeados en Python.
"""

import os
from functools import lru_cache
from typing import Dict, List, Tuple

import yaml

from src.models.schemas import (
    RatiosFinancieros,
    PoliticasEvaluacion,
    InfoCrediticia,
    InfoTributaria,
    TipoProducto,
    EstadoEvaluacion,
)

# ---------------------------------------------------------------------------
# Ruta al archivo de política crediticia
# ---------------------------------------------------------------------------
_POLICY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "data", "politica_credito.yaml",
)


def _policy_path() -> str:
    """Retorna la ruta absoluta a ``data/politica_credito.yaml``."""
    return os.path.normpath(_POLICY_PATH)


# ---------------------------------------------------------------------------
# Caché de la configuración YAML (se carga una sola vez)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def cargar_politica() -> dict:
    """Carga y retorna el contenido completo de ``politica_credito.yaml``.

    El resultado queda cacheado por ``lru_cache`` para evitar múltiples
    lecturas de disco durante una misma sesión.
    """
    ruta = _policy_path()
    if not os.path.isfile(ruta):
        raise FileNotFoundError(
            f"No se encuentra el archivo de política crediticia: {ruta}"
        )
    with open(ruta, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _get(config: dict, *keys, default=None):
    """Navega por un diccionario anidado siguiendo *keys.

    Ejemplo::

        _get(config, "scoring", "penalizaciones", "protestos")  # → 20
        _get(config, "inexistente", default=0)                   # → 0
    """
    current = config
    for k in keys:
        if isinstance(current, dict):
            current = current.get(k)
            if current is None:
                return default
        else:
            return default
    return current if current is not None else default


# ---------------------------------------------------------------------------
# Funciones auxiliares de extracción de umbrales
# ---------------------------------------------------------------------------

def _ratio_umbral(config: dict, ratio_name: str, campo: str) -> float:
    """Obtiene un umbral numérico de ``config["ratios_financieros"]``."""
    return float(
        _get(config, "ratios_financieros", ratio_name, campo, default=0.0)
    )


def _penalizacion(config: dict, nombre: str) -> float:
    """Obtiene un valor de penalización desde ``scoring.penalizaciones``."""
    return float(
        _get(config, "scoring", "penalizaciones", nombre, default=0)
    )


def _bonificacion(config: dict, nombre: str) -> float:
    """Obtiene un valor de bonificación desde ``scoring.bonificaciones``."""
    return float(
        _get(config, "scoring", "bonificaciones", nombre, default=0)
    )


# ===========================================================================
# EVALUACIÓN PRINCIPAL
# ===========================================================================


def evaluar_cumplimiento(
    ratios: RatiosFinancieros,
    politicas: PoliticasEvaluacion,
    info_crediticia: InfoCrediticia,
    info_tributaria: InfoTributaria,
) -> Tuple[List[str], float]:
    """Evalúa el cumplimiento de políticas crediticias.

    Analiza cada política configurada contra los ratios calculados
    y la información crediticia, retornando los incumplimientos
    y un puntaje de riesgo.

    Args:
        ratios: Ratios financieros calculados.
        politicas: Políticas de evaluación configuradas.
        info_crediticia: Información crediticia del deudor.
        info_tributaria: Información tributaria de la empresa.

    Returns:
        Tupla con (lista de incumplimientos, puntaje de riesgo 0-100).
    """
    config = cargar_politica()
    incumplimientos: List[str] = []
    puntaje = float(
        _get(config, "scoring", "puntaje_base_neutral", default=50.0)
    )

    # --- Validación de antigüedad ---
    if info_crediticia.antiguedad_meses < politicas.antiguedad_minima_meses:
        incumplimientos.append(
            f"Antigüedad insuficiente: {info_crediticia.antiguedad_meses} meses "
            f"(mínimo {politicas.antiguedad_minima_meses} meses)"
        )
        puntaje += _penalizacion(config, "antiguedad_insuficiente")

    # --- Validación de liquidez ---
    if ratios.liquidez_corriente < politicas.liquidez_minima:
        incumplimientos.append(
            f"Liquidez corriente insuficiente: {ratios.liquidez_corriente:.2f} "
            f"(mínimo {politicas.liquidez_minima:.2f})"
        )
        puntaje += _penalizacion(config, "liquidez_baja")
    elif ratios.liquidez_corriente >= politicas.liquidez_minima * 1.5:
        puntaje += _bonificacion(config, "liquidez_solida")

    # --- Validación de endeudamiento ---
    if ratios.endeudamiento_total > politicas.endeudamiento_maximo:
        incumplimientos.append(
            f"Endeudamiento excesivo: {ratios.endeudamiento_total:.2f} "
            f"(máximo {politicas.endeudamiento_maximo:.2f})"
        )
        puntaje += _penalizacion(config, "endeudamiento_alto")
    elif ratios.endeudamiento_total <= politicas.endeudamiento_maximo * 0.5:
        puntaje += _bonificacion(config, "endeudamiento_bajo")

    # --- Validación deuda/patrimonio ---
    umbral_dp = _ratio_umbral(config, "razon_deuda_patrimonio_maximo", "umbral_maximo")
    if ratios.razon_deuda_patrimonio > umbral_dp:
        incumplimientos.append(
            f"Alta dependencia de deuda: razón deuda/patrimonio de "
            f"{ratios.razon_deuda_patrimonio:.2f} (máximo {umbral_dp:.2f})"
        )
        puntaje += _penalizacion(config, "deuda_patrimonio_alta")

    # --- Validación específica para leasing ---
    if politicas.producto == TipoProducto.LEASING:
        if ratios.cobertura_servicio_deuda_fcf < politicas.cobertura_ebitda_minima:
            incumplimientos.append(
                f"Cobertura EBITDA insuficiente para leasing: "
                f"{ratios.cobertura_servicio_deuda_fcf:.2f} "
                f"(mínimo {politicas.cobertura_ebitda_minima:.2f})"
            )
            puntaje += _penalizacion(config, "cobertura_ebitda_insuficiente")
        elif ratios.cobertura_servicio_deuda_fcf >= politicas.cobertura_ebitda_minima * 1.5:
            puntaje += _bonificacion(config, "cobertura_ebitda_solida")
    else:
        # Para factoring, validar rotación de cartera
        umbral_rotacion = _ratio_umbral(config, "rotacion_cartera_minima", "umbral_minimo")
        if ratios.rotacion_cartera < umbral_rotacion:
            incumplimientos.append(
                f"Baja rotación de cartera para factoring: "
                f"{ratios.rotacion_cartera:.2f} (mínimo {umbral_rotacion:.2f})"
            )
            puntaje += _penalizacion(config, "rotacion_cartera_baja")

    # --- Validación crediticia ---
    if info_crediticia.tiene_protestos:
        incumplimientos.append(
            f"Registra protestos por ${info_crediticia.monto_protestos:,.0f}"
        )
        puntaje += _penalizacion(config, "protestos")

    if info_crediticia.tiene_morosidades:
        if not politicas.morosidades_permitidas:
            incumplimientos.append("Registra morosidades (no permitidas)")
            puntaje += _penalizacion(config, "morosidad_no_permitida")
        elif info_crediticia.dias_morosidad > politicas.dias_morosidad_maximo:
            incumplimientos.append(
                f"Días de morosidad exceden máximo: "
                f"{info_crediticia.dias_morosidad} días "
                f"(máximo {politicas.dias_morosidad_maximo})"
            )
            puntaje += _penalizacion(config, "morosidad_dias_excedidos")

    if info_crediticia.morosidades_previsionales:
        incumplimientos.append("Registra morosidades previsionales")
        puntaje += _penalizacion(config, "morosidades_previsionales")

    # --- Validación de margen EBITDA ---
    umbral_margen_min = (
        float(_get(config, "ratios_financieros", "margen_ebitda", "umbral_minimo_pct", "valor", default=5))
        / 100.0
    )
    umbral_margen_bonif = (
        float(_get(config, "ratios_financieros", "margen_ebitda", "umbral_bonificacion_pct", "valor", default=20))
        / 100.0
    )

    if ratios.margen_ebitda < umbral_margen_min:
        incumplimientos.append(
            f"Margen EBITDA muy bajo: {ratios.margen_ebitda:.2%}"
        )
        puntaje += _penalizacion(config, "margen_ebitda_bajo")
    elif ratios.margen_ebitda >= umbral_margen_bonif:
        puntaje += _bonificacion(config, "margen_ebitda_alto")

    # --- ROI ---
    umbral_roi = _ratio_umbral(config, "roi_minimo", "umbral_minimo")
    if ratios.roi < umbral_roi:
        incumplimientos.append("ROI negativo")
        puntaje += _penalizacion(config, "roi_negativo")

    # Normalizar puntaje al rango 0-100
    puntaje = max(0.0, min(100.0, puntaje))

    return incumplimientos, puntaje


# ===========================================================================
# DETERMINACIÓN DE ESTADO
# ===========================================================================


def determinar_estado(
    incumplimientos: List[str],
    puntaje_riesgo: float
) -> EstadoEvaluacion:
    """Determina el estado de evaluación basado en incumplimientos y puntaje.

    Args:
        incumplimientos: Lista de políticas incumplidas.
        puntaje_riesgo: Puntaje de riesgo calculado (0-100).

    Returns:
        Estado de evaluación correspondiente.
    """
    config = cargar_politica()

    umbral_aprobado = float(
        _get(config, "scoring", "umbrales_decision", "aprobado", default=40)
    )
    umbral_aprobado_cond_max_inc = int(
        _get(
            config,
            "scoring", "umbrales_decision",
            "aprobado_con_condiciones", "max_incumplimientos",
            default=2,
        )
    )
    umbral_aprobado_cond_max_score = float(
        _get(
            config,
            "scoring", "umbrales_decision",
            "aprobado_con_condiciones", "puntaje_maximo",
            default=60,
        )
    )
    umbral_rechazado = float(
        _get(config, "scoring", "umbrales_decision", "rechazado", default=80)
    )

    if not incumplimientos and puntaje_riesgo < umbral_aprobado:
        return EstadoEvaluacion.APROBADO
    elif (
        len(incumplimientos) <= umbral_aprobado_cond_max_inc
        and puntaje_riesgo < umbral_aprobado_cond_max_score
    ):
        return EstadoEvaluacion.APROBADO_CON_CONDICIONES
    elif puntaje_riesgo >= umbral_rechazado:
        return EstadoEvaluacion.RECHAZADO
    else:
        return EstadoEvaluacion.REQUIERE_ANALISIS_MANUAL