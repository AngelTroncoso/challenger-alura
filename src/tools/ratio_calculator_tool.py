"""Cálculo de ratios financieros para evaluación crediticia chilena.

Implementa fórmulas estándar para factoring y leasing, retornando
siempre un objeto ``RatiosFinancieros`` definido en ``schemas.py``.
"""

from typing import Optional

from src.models.schemas import (
    EstadosFinancieros,
    InfoCrediticia,
    RatiosFinancieros,
)


# ---------------------------------------------------------------------------
# Ratios de Liquidez
# ---------------------------------------------------------------------------


def _calcular_razon_corriente(ee: EstadosFinancieros) -> float:
    """Razón Corriente = Activo Corriente / Pasivo Corriente

    Indica la capacidad de la empresa para cubrir obligaciones de corto
    plazo con activos de corto plazo.
    """
    if ee.pasivos_corrientes == 0:
        return 0.0
    return round(ee.activos_corrientes / ee.pasivos_corrientes, 4)


def _calcular_prueba_acida(ee: EstadosFinancieros) -> float:
    """Prueba Ácida = (Activo Corriente - Inventarios) / Pasivo Corriente

    Como no tenemos inventarios directamente, estimamos el inventario como
    ``Costo de Ventas / 12`` (rotación mensual aproximada).

    Si no hay costo de ventas, se retorna la razón corriente como proxy.
    """
    if ee.pasivos_corrientes == 0:
        return 0.0
    if ee.costo_ventas > 0:
        inventario_estimado = ee.costo_ventas / 12.0
    else:
        inventario_estimado = 0.0
    activo_liquido = max(0.0, ee.activos_corrientes - inventario_estimado)
    return round(activo_liquido / ee.pasivos_corrientes, 4)


# ---------------------------------------------------------------------------
# Ratios de Endeudamiento
# ---------------------------------------------------------------------------


def _calcular_endeudamiento_total(ee: EstadosFinancieros) -> float:
    """Endeudamiento Total = Pasivo Total / Activo Total

    Porcentaje de los activos financiados por deuda.
    """
    activo_total = ee.activos_corrientes + ee.activos_no_corrientes
    if activo_total == 0:
        return 0.0
    pasivo_total = ee.pasivos_corrientes + ee.pasivos_no_corrientes
    return round(pasivo_total / activo_total, 4)


def _calcular_razon_deuda_patrimonio(ee: EstadosFinancieros) -> float:
    """Razón Deuda / Patrimonio = Pasivo Total / Patrimonio

    Mide la proporción de financiamiento externo vs. capital propio.
    """
    if ee.patrimonio == 0:
        return 0.0
    pasivo_total = ee.pasivos_corrientes + ee.pasivos_no_corrientes
    return round(pasivo_total / ee.patrimonio, 4)


# ---------------------------------------------------------------------------
# Ratios de Cobertura (Leasing)
# ---------------------------------------------------------------------------


def _calcular_ebitda(ee: EstadosFinancieros) -> float:
    """EBITDA = Resultado Operacional + Depreciación"""
    return ee.resultado_operacional + ee.depreciacion


def _calcular_margen_ebitda(ee: EstadosFinancieros) -> float:
    """Margen EBITDA = EBITDA / Ventas Netas"""
    if ee.ventas_netas == 0:
        return 0.0
    ebitda = _calcular_ebitda(ee)
    return round(ebitda / ee.ventas_netas, 4)


def _calcular_cobertura_ebitda(ee: EstadosFinancieros) -> float:
    """Cobertura EBITDA = EBITDA / Gastos Financieros

    Mide cuántas veces el EBITDA cubre los gastos financieros.
    """
    if ee.gastos_financieros == 0:
        # Si no hay gastos financieros, la cobertura es alta (señal positiva)
        # Retornamos un valor alto > umbral para evitar falsos rechazos
        return 10.0
    ebitda = _calcular_ebitda(ee)
    return round(ebitda / ee.gastos_financieros, 4)


def _calcular_cobertura_fcf(ee: EstadosFinancieros) -> float:
    """Cobertura Flujo de Caja Libre = (EBITDA - Capex estimado) / Gastos Financieros

    Para leasing, el FCF se aproxima como EBITDA - 30% EBITDA (capex estimado).
    Si no hay gastos financieros, se retorna un valor alto.
    """
    if ee.gastos_financieros == 0:
        return 10.0
    ebitda = _calcular_ebitda(ee)
    capex_estimado = ebitda * 0.3  # 30% del EBITDA como capex estimado
    fcf = max(0.0, ebitda - capex_estimado)
    return round(fcf / ee.gastos_financieros, 4)


# ---------------------------------------------------------------------------
# Ratios de Factoring (Rotación de Cartera)
# ---------------------------------------------------------------------------


def _estimar_ctas_cobrar(ee: EstadosFinancieros) -> float:
    """Estima Cuentas por Cobrar a partir de activos corrientes.

    Se asume que ~40% del activo corriente son cuentas por cobrar
    (proporción típica en empresas chilenas). Si hay información
    crediticia con rotación, se refina después.
    """
    return ee.activos_corrientes * 0.40


def _calcular_rotacion_cartera(
    ee: EstadosFinancieros,
    info_cred: Optional[InfoCrediticia] = None,
) -> float:
    """Rotación de Cuentas por Cobrar (veces) = Ventas Netas / Cuentas por Cobrar

    Mide la eficiencia en la cobranza. Mayor rotación indica mejor
    gestión de crédito.
    """
    ctas_cobrar = _estimar_ctas_cobrar(ee)
    if ctas_cobrar == 0:
        return 0.0
    return round(ee.ventas_netas / ctas_cobrar, 4)


def _calcular_dias_calle_promedio(
    ee: EstadosFinancieros,
    info_cred: Optional[InfoCrediticia] = None,
) -> float:
    """Días Calle Promedio = 365 / Rotación de Cartera

    Días promedio que tarda la empresa en cobrar sus facturas.
    """
    rotacion = _calcular_rotacion_cartera(ee, info_cred)
    if rotacion == 0:
        return 0.0
    return round(365.0 / rotacion, 2)


# ---------------------------------------------------------------------------
# ROI
# ---------------------------------------------------------------------------


def _calcular_roi(ee: EstadosFinancieros) -> float:
    """ROI = Resultado Neto / Activo Total

    Retorno sobre la inversión total.
    """
    activo_total = ee.activos_corrientes + ee.activos_no_corrientes
    if activo_total == 0:
        return 0.0
    return round(ee.resultado_neto / activo_total, 4)


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------


def calcular_ratios_factoring(
    ee: EstadosFinancieros,
    info_cred: Optional[InfoCrediticia] = None,
) -> RatiosFinancieros:
    """Calcula todos los ratios financieros relevantes para factoring.

    Para factoring se priorizan: liquidez (corto plazo), rotación de
    cartera, endeudamiento y rentabilidad.

    Args:
        ee: Estados financieros de la empresa.
        info_cred: Información crediticia (opcional, mejora estimación).

    Returns:
        Objeto ``RatiosFinancieros`` con todos los ratios calculados.
    """
    return RatiosFinancieros(
        liquidez_corriente=_calcular_razon_corriente(ee),
        liquidez_inmediata=_calcular_prueba_acida(ee),
        razon_deuda_patrimonio=_calcular_razon_deuda_patrimonio(ee),
        endeudamiento_total=_calcular_endeudamiento_total(ee),
        cobertura_servicio_deuda_fcf=_calcular_cobertura_fcf(ee),
        margen_ebitda=_calcular_margen_ebitda(ee),
        roi=_calcular_roi(ee),
        rotacion_cartera=_calcular_rotacion_cartera(ee, info_cred),
    )


def calcular_ratios_leasing(
    ee: EstadosFinancieros,
    info_cred: Optional[InfoCrediticia] = None,
) -> RatiosFinancieros:
    """Calcula todos los ratios financieros relevantes para leasing.

    Para leasing se priorizan: cobertura EBITDA, capacidad de pago,
    endeudamiento y márgenes.

    Args:
        ee: Estados financieros de la empresa.
        info_cred: Información crediticia (opcional, no se usa directamente
                   en leasing pero se incluye para firma unificada).

    Returns:
        Objeto ``RatiosFinancieros`` con todos los ratios calculados.
    """
    return RatiosFinancieros(
        liquidez_corriente=_calcular_razon_corriente(ee),
        liquidez_inmediata=_calcular_prueba_acida(ee),
        razon_deuda_patrimonio=_calcular_razon_deuda_patrimonio(ee),
        endeudamiento_total=_calcular_endeudamiento_total(ee),
        cobertura_servicio_deuda_fcf=_calcular_cobertura_fcf(ee),
        margen_ebitda=_calcular_margen_ebitda(ee),
        roi=_calcular_roi(ee),
        rotacion_cartera=0.0,  # No aplica para leasing
    )