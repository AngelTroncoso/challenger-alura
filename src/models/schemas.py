"""Modelos Pydantic para el dominio de evaluación crediticia chilena."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Literal
from datetime import date
from enum import Enum


class TipoProducto(str, Enum):
    FACTORING = "factoring"
    LEASING = "leasing"


class EstadoEvaluacion(str, Enum):
    APROBADO = "aprobado"
    APROBADO_CON_CONDICIONES = "aprobado_con_condiciones"
    RECHAZADO = "rechazado"
    REQUIERE_ANALISIS_MANUAL = "requiere_analisis_manual"


class InfoTributaria(BaseModel):
    """Información tributaria extraída del SII (Formularios 29 y 22)."""
    rut_empresa: str = Field(..., description="RUT de la empresa sin puntos con guión")
    razon_social: str = Field(..., description="Razón social completa")
    fecha_inicio_actividades: Optional[date] = Field(None, description="Fecha de inicio de actividades en SII")
    formulario29_ultimos_12m: List[Dict[str, float]] = Field(
        default_factory=list,
        description="Lista mensual con IVA ventas, IVA compras, impuesto pagado"
    )
    formulario22_anual: Dict[str, float] = Field(
        default_factory=dict,
        description="Renta líquida, impuesto pagado, capital propio"
    )


class EstadosFinancieros(BaseModel):
    """Estados financieros de la empresa (balance y resultado)."""
    periodo: str = Field(..., description="Período contable (ej: '2024-12')")
    activos_corrientes: float = Field(0.0, ge=0)
    activos_no_corrientes: float = Field(0.0, ge=0)
    pasivos_corrientes: float = Field(0.0, ge=0)
    pasivos_no_corrientes: float = Field(0.0, ge=0)
    patrimonio: float = Field(0.0, ge=0)
    ventas_netas: float = Field(0.0, ge=0)
    costo_ventas: float = Field(0.0, ge=0)
    resultado_operacional: float = Field(0.0)
    resultado_neto: float = Field(0.0)
    depreciacion: float = Field(0.0, ge=0)
    gastos_financieros: float = Field(0.0, ge=0)


class InfoCrediticia(BaseModel):
    """Información crediticia y comercial del deudor."""
    tiene_protestos: bool = False
    monto_protestos: float = 0.0
    tiene_morosidades: bool = False
    dias_morosidad: int = 0
    morosidades_previsionales: bool = False
    antiguedad_meses: int = Field(0, ge=0, description="Antigüedad de la empresa en meses")


class RatiosFinancieros(BaseModel):
    """Ratios financieros calculados a partir de estados financieros."""
    liquidez_corriente: float = Field(0.0, description="Activo corriente / Pasivo corriente")
    liquidez_inmediata: float = Field(0.0, description="(Activo corriente - Inventarios) / Pasivo corriente")
    razon_deuda_patrimonio: float = Field(0.0, description="Pasivo total / Patrimonio")
    endeudamiento_total: float = Field(0.0, description="Pasivo total / Activo total")
    cobertura_servicio_deuda_fcf: float = Field(
        0.0, description="(EBITDA - Capex) / Servicio deuda - para leasing"
    )
    margen_ebitda: float = Field(0.0, description="EBITDA / Ventas netas")
    roi: float = Field(0.0, description="Resultado neto / Activo total")
    rotacion_cartera: float = Field(0.0, description="Ventas netas / Cuentas por cobrar - para factoring")


class PoliticasEvaluacion(BaseModel):
    """Parámetros de política de evaluación crediticia configurables."""
    producto: TipoProducto = TipoProducto.FACTORING
    antiguedad_minima_meses: int = Field(12, ge=0, description="Antigüedad mínima requerida")
    liquidez_minima: float = Field(1.2, ge=0, description="Razón circulante mínima")
    endeudamiento_maximo: float = Field(2.5, ge=0, description="Endeudamiento máximo permitido")
    cobertura_ebitda_minima: float = Field(1.5, ge=0, description="Cobertura EBITDA mínima (leasing)")
    morosidades_permitidas: bool = Field(False, description="Si se permiten morosidades")
    dias_morosidad_maximo: int = Field(0, ge=0, description="Días máximos de morosidad permitidos")


class ResultadoEvaluacion(BaseModel):
    """Resultado completo de la evaluación crediticia."""
    rut_empresa: str
    producto: TipoProducto
    estado: EstadoEvaluacion
    puntaje_riesgo: float = Field(..., ge=0, le=100, description="Puntaje de riesgo 0-100")
    ratios_calculados: RatiosFinancieros
    politicas_aplicadas: PoliticasEvaluacion
    dictamen: str = Field(..., description="Dictamen textual de la evaluación")
    condiciones_aplicables: List[str] = Field(default_factory=list)
    factores_positivos: List[str] = Field(default_factory=list)
    factores_riesgo: List[str] = Field(default_factory=list)
    recomendaciones: List[str] = Field(default_factory=list)