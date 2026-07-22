"""Estado compartido del grafo LangGraph para el flujo multiagente.

Define el ``GraphState`` (TypedDict) que fluye a través del grafo,
y las funciones nodo que conectan a los 4 agentes especializados:

1. **IngestorAgent** → extrae datos desde PDFs
2. **AnalistaFinancieroAgent** → calcula ratios financieros
3. **EvaluadorRiesgoAgent** → aplica políticas crediticias
4. **DictaminadorAgent** → genera dictamen final
"""

import logging
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from src.models.schemas import (
    EstadosFinancieros,
    EstadoEvaluacion,
    InfoCrediticia,
    InfoTributaria,
    PoliticasEvaluacion,
    RatiosFinancieros,
    ResultadoEvaluacion,
    TipoProducto,
)
from src.tools.policy_engine_tool import (
    determinar_estado,
    evaluar_cumplimiento,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# GraphState — estado compartido del grafo
# ===========================================================================


class GraphState(TypedDict):
    """Estado compartido que fluye a través del grafo multiagente.

    Cada agente lee y escribe campos específicos de este estado,
    permitiendo la comunicación entre etapas del proceso.
    """

    # Metadata del proceso
    pdf_path: str
    producto: TipoProducto
    rut_empresa: Optional[str]
    razon_social: Optional[str]

    # Documentos extraídos (salida del IngestorAgent)
    info_tributaria: Optional[InfoTributaria]
    estados_financieros: Optional[EstadosFinancieros]
    info_crediticia: Optional[InfoCrediticia]

    # Análisis financiero (salida del AnalistaFinancieroAgent)
    ratios_calculados: Optional[RatiosFinancieros]

    # Políticas y evaluación (salida del EvaluadorRiesgoAgent)
    politicas_aplicadas: Optional[PoliticasEvaluacion]
    incumplimientos: List[str]
    puntaje_riesgo: float

    # Dictamen final (salida del DictaminadorAgent)
    resultado: Optional[ResultadoEvaluacion]

    # Metadatos del proceso
    errores: List[str]
    estado_actual: str  # 'ingestion' | 'analisis' | 'evaluacion' | 'dictamen' | 'completo' | 'error'


# ===========================================================================
# Estado inicial por defecto
# ===========================================================================

DEFAULT_POLITICAS = PoliticasEvaluacion()


def estado_inicial(
    pdf_path: str,
    producto: TipoProducto = TipoProducto.FACTORING,
) -> GraphState:
    """Construye un estado inicial para comenzar el flujo.

    Args:
        pdf_path: Ruta al archivo PDF del dossier.
        producto: Tipo de producto (factoring o leasing).

    Returns:
        ``GraphState`` con valores por defecto.
    """
    return GraphState(
        pdf_path=pdf_path,
        producto=producto,
        rut_empresa=None,
        razon_social=None,
        info_tributaria=None,
        estados_financieros=None,
        info_crediticia=None,
        ratios_calculados=None,
        politicas_aplicadas=DEFAULT_POLITICAS.model_copy(
            update={"producto": producto}
        ),
        incumplimientos=[],
        puntaje_riesgo=50.0,
        resultado=None,
        errores=[],
        estado_actual="ingestion",
    )


# ===========================================================================
# Nodos del grafo (funciones que procesan el estado)
# ===========================================================================


def nodo_ingestion(state: GraphState) -> Dict[str, Any]:
    """Nodo de ingesta: ejecuta el IngestorAgent sobre el PDF.

    Lee ``pdf_path`` del estado, ejecuta el agente y extrae
    ``info_tributaria``, ``estados_financieros`` e ``info_crediticia``.

    Args:
        state: Estado actual del grafo.

    Returns:
        Diccionario con las actualizaciones al estado.
    """
    logger.info("=== NODO INGESTIÓN ===")
    pdf_path = state.get("pdf_path", "")
    if not pdf_path:
        return {
            "errores": ["No se especificó ruta de PDF"],
            "estado_actual": "error",
        }

    try:
        # Usar el parser directo en lugar del agente LangChain para evitar
        # dependencia de LLM en esta etapa (el parser es puramente local)
        from src.tools.pdf_parser_tool import procesar_dossier

        dossier = procesar_dossier(pdf_path)

        if dossier.get("error"):
            return {
                "errores": [dossier["error"]],
                "estado_actual": "error",
            }

        info_trib = dossier.get("info_tributaria")
        ee = dossier.get("estados_financieros")
        info_cred = dossier.get("info_crediticia")

        actualizacion: Dict[str, Any] = {
            "info_tributaria": info_trib,
            "estados_financieros": ee,
            "info_crediticia": info_cred,
            "estado_actual": "analisis",
        }

        if info_trib:
            actualizacion["rut_empresa"] = info_trib.rut_empresa
            actualizacion["razon_social"] = info_trib.razon_social

        return actualizacion

    except Exception as exc:
        logger.error("Error en nodo_ingestion: %s", exc)
        return {
            "errores": [f"Error en ingesta: {exc}"],
            "estado_actual": "error",
        }


def nodo_analisis(state: GraphState) -> Dict[str, Any]:
    """Nodo de análisis financiero: calcula ratios.

    Toma ``estados_financieros`` e ``info_crediticia`` del estado
    y calcula los ratios financieros según el tipo de producto.

    Args:
        state: Estado actual del grafo.

    Returns:
        Diccionario con ``ratios_calculados`` actualizado.
    """
    logger.info("=== NODO ANÁLISIS FINANCIERO ===")
    ee = state.get("estados_financieros")
    info_cred = state.get("info_crediticia") or InfoCrediticia()
    producto = state.get("producto", TipoProducto.FACTORING)

    if ee is None:
        return {
            "errores": ["No hay estados financieros para analizar"],
            "estado_actual": "error",
        }

    try:
        from src.tools.ratio_calculator_tool import (
            calcular_ratios_factoring,
            calcular_ratios_leasing,
        )

        if producto == TipoProducto.LEASING:
            ratios = calcular_ratios_leasing(ee, info_cred)
        else:
            ratios = calcular_ratios_factoring(ee, info_cred)

        return {
            "ratios_calculados": ratios,
            "estado_actual": "evaluacion",
        }

    except Exception as exc:
        logger.error("Error en nodo_analisis: %s", exc)
        return {
            "errores": [f"Error en análisis financiero: {exc}"],
            "estado_actual": "error",
        }


def nodo_evaluacion(state: GraphState) -> Dict[str, Any]:
    """Nodo de evaluación de riesgo: aplica políticas crediticias.

    Toma los ratios calculados, la información crediticia y tributaria,
    y aplica las políticas configuradas para determinar incumplimientos,
    puntaje de riesgo y estado de evaluación.

    Args:
        state: Estado actual del grafo.

    Returns:
        Diccionario con ``incumplimientos``, ``puntaje_riesgo``
        y ``politicas_aplicadas`` actualizados.
    """
    logger.info("=== NODO EVALUACIÓN DE RIESGO ===")
    ratios = state.get("ratios_calculados")
    info_cred = state.get("info_crediticia") or InfoCrediticia()
    info_trib = state.get("info_tributaria") or InfoTributaria(
        rut_empresa=state.get("rut_empresa") or "",
        razon_social=state.get("razon_social") or "",
    )
    politicas = state.get("politicas_aplicadas") or DEFAULT_POLITICAS

    if ratios is None:
        return {
            "errores": ["No hay ratios financieros para evaluar"],
            "estado_actual": "error",
        }

    try:
        incumplimientos, puntaje = evaluar_cumplimiento(
            ratios, politicas, info_cred, info_trib
        )
        estado = determinar_estado(incumplimientos, puntaje)

        return {
            "incumplimientos": incumplimientos,
            "puntaje_riesgo": puntaje,
            "politicas_aplicadas": politicas,
            "estado_actual": "dictamen",
        }

    except Exception as exc:
        logger.error("Error en nodo_evaluacion: %s", exc)
        return {
            "errores": [f"Error en evaluación de riesgo: {exc}"],
            "estado_actual": "error",
        }


def nodo_dictamen(state: GraphState) -> Dict[str, Any]:
    """Nodo de dictamen: genera el resultado final estructurado.

    Consolida toda la información del proceso en un
    ``ResultadoEvaluacion`` con dictamen textual, condiciones,
    factores de riesgo y recomendaciones.

    Args:
        state: Estado actual del grafo.

    Returns:
        Diccionario con ``resultado`` y ``estado_actual``.
    """
    logger.info("=== NODO DICTAMEN ===")

    try:
        info_trib = state.get("info_tributaria") or InfoTributaria(
            rut_empresa=state.get("rut_empresa") or "",
            razon_social=state.get("razon_social") or "",
        )
        ee = state.get("estados_financieros")
        info_cred = state.get("info_crediticia") or InfoCrediticia()
        ratios = state.get("ratios_calculados") or RatiosFinancieros()
        politicas = state.get("politicas_aplicadas") or DEFAULT_POLITICAS
        incumplimientos = state.get("incumplimientos", [])
        puntaje = state.get("puntaje_riesgo", 50.0)

        estado = determinar_estado(incumplimientos, puntaje)

        # Generar dictamen textual estructurado
        dictamen = _generar_dictamen_texto(
            info_trib=info_trib,
            ee=ee,
            info_cred=info_cred,
            ratios=ratios,
            politicas=politicas,
            incumplimientos=incumplimientos,
            puntaje=puntaje,
            estado=estado,
        )

        # Factores positivos y de riesgo
        factores_positivos = _identificar_factores_positivos(ratios, politicas)
        factores_riesgo = _identificar_factores_riesgo(
            incumplimientos, info_cred
        )

        # Condiciones aplicables
        condiciones = _generar_condiciones(estado, incumplimientos, politicas)

        # Recomendaciones
        recomendaciones = _generar_recomendaciones(
            estado, incumplimientos, ratios, info_cred
        )

        resultado = ResultadoEvaluacion(
            rut_empresa=info_trib.rut_empresa or state.get("rut_empresa") or "",
            producto=politicas.producto,
            estado=estado,
            puntaje_riesgo=puntaje,
            ratios_calculados=ratios,
            politicas_aplicadas=politicas,
            dictamen=dictamen,
            condiciones_aplicables=condiciones,
            factores_positivos=factores_positivos,
            factores_riesgo=factores_riesgo,
            recomendaciones=recomendaciones,
        )

        return {
            "resultado": resultado,
            "estado_actual": "completo",
        }

    except Exception as exc:
        logger.error("Error en nodo_dictamen: %s", exc)
        return {
            "errores": [f"Error generando dictamen: {exc}"],
            "estado_actual": "error",
        }


# ===========================================================================
# Funciones auxiliares de generación de texto
# ===========================================================================


def _generar_dictamen_texto(
    info_trib: InfoTributaria,
    ee: Optional[EstadosFinancieros],
    info_cred: InfoCrediticia,
    ratios: RatiosFinancieros,
    politicas: PoliticasEvaluacion,
    incumplimientos: List[str],
    puntaje: float,
    estado: EstadoEvaluacion,
) -> str:
    """Genera el texto del dictamen crediticio.

    Args:
        info_trib: Información tributaria de la empresa.
        ee: Estados financieros (opcional).
        info_cred: Información crediticia.
        ratios: Ratios financieros calculados.
        politicas: Políticas aplicadas.
        incumplimientos: Lista de incumplimientos detectados.
        puntaje: Puntaje de riesgo (0-100).
        estado: Estado de la evaluación.

    Returns:
        Texto del dictamen en formato informe.
    """
    nombre_empresa = info_trib.razon_social or "No especificada"
    rut = info_trib.rut_empresa or "No especificado"
    producto_str = "Factoring" if politicas.producto == TipoProducto.FACTORING else "Leasing"

    lineas = [
        f"DICTAMEN DE EVALUACIÓN CREDITICIA — {producto_str}",
        f"{'=' * 60}",
        "",
        f"Empresa: {nombre_empresa}",
        f"RUT: {rut}",
        f"Producto solicitado: {producto_str}",
        f"Estado: {estado.value.upper()}",
        f"Puntaje de riesgo: {puntaje:.1f}/100",
        "",
    ]

    if incumplimientos:
        lineas.append("Incumplimientos detectados:")
        for i, inc in enumerate(incumplimientos, 1):
            lineas.append(f"  {i}. {inc}")
        lineas.append("")

    # Resumen financiero
    lineas.append("Resumen de ratios financieros:")
    lineas.append(f"  Liquidez corriente: {ratios.liquidez_corriente:.2f}")
    lineas.append(f"  Endeudamiento total: {ratios.endeudamiento_total:.2%}")
    lineas.append(f"  Margen EBITDA: {ratios.margen_ebitda:.2%}")
    lineas.append(f"  ROI: {ratios.roi:.2%}")

    if politicas.producto == TipoProducto.FACTORING:
        lineas.append(f"  Rotación de cartera: {ratios.rotacion_cartera:.2f} veces")
    else:
        lineas.append(
            f"  Cobertura FCF: {ratios.cobertura_servicio_deuda_fcf:.2f}x"
        )

    lineas.append("")

    # Conclusión
    if estado == EstadoEvaluacion.APROBADO:
        lineas.append(
            "CONCLUSIÓN: El solicitante cumple con todos los requisitos "
            "de la política crediticia. Se recomienda aprobación."
        )
    elif estado == EstadoEvaluacion.APROBADO_CON_CONDICIONES:
        lineas.append(
            "CONCLUSIÓN: El solicitante cumple parcialmente los requisitos. "
            "Se recomienda aprobación sujeta a condiciones."
        )
    elif estado == EstadoEvaluacion.RECHAZADO:
        lineas.append(
            "CONCLUSIÓN: El solicitante no cumple con los requisitos mínimos "
            "de la política crediticia. Se recomienda rechazo."
        )
    else:
        lineas.append(
            "CONCLUSIÓN: Se requiere análisis manual adicional debido a "
            "la complejidad del caso."
        )

    return "\n".join(lineas)


def _identificar_factores_positivos(
    ratios: RatiosFinancieros,
    politicas: PoliticasEvaluacion,
) -> List[str]:
    """Identifica factores positivos en la evaluación.

    Args:
        ratios: Ratios financieros calculados.
        politicas: Políticas aplicadas.

    Returns:
        Lista de factores positivos.
    """
    factores: List[str] = []

    if ratios.liquidez_corriente >= politicas.liquidez_minima * 1.5:
        factores.append("Sólida posición de liquidez")
    if ratios.endeudamiento_total <= politicas.endeudamiento_maximo * 0.5:
        factores.append("Bajo nivel de endeudamiento")
    if ratios.margen_ebitda >= 0.20:
        factores.append("Margen EBITDA saludable (>20%)")
    if ratios.roi > 0.10:
        factores.append("ROI atractivo (>10%)")
    if ratios.cobertura_servicio_deuda_fcf >= 2.0:
        factores.append("Fuerte capacidad de cobertura de deuda")

    return factores


def _identificar_factores_riesgo(
    incumplimientos: List[str],
    info_cred: InfoCrediticia,
) -> List[str]:
    """Identifica factores de riesgo en la evaluación.

    Args:
        incumplimientos: Lista de incumplimientos detectados.
        info_cred: Información crediticia.

    Returns:
        Lista de factores de riesgo.
    """
    factores: List[str] = list(incumplimientos)

    if info_cred.tiene_protestos:
        factores.append("Registro de protestos vigentes")
    if info_cred.tiene_morosidades:
        factores.append(f"Morosidad registrada ({info_cred.dias_morosidad} días)")
    if info_cred.morosidades_previsionales:
        factores.append("Morosidades previsionales pendientes")

    return factores


def _generar_condiciones(
    estado: EstadoEvaluacion,
    incumplimientos: List[str],
    politicas: PoliticasEvaluacion,
) -> List[str]:
    """Genera condiciones aplicables según el estado de evaluación.

    Args:
        estado: Estado de la evaluación.
        incumplimientos: Lista de incumplimientos.
        politicas: Políticas aplicadas.

    Returns:
        Lista de condiciones.
    """
    if estado == EstadoEvaluacion.APROBADO:
        return ["Sujeción a políticas generales de crédito"]

    if estado == EstadoEvaluacion.APROBADO_CON_CONDICIONES:
        condiciones: List[str] = []
        for inc in incumplimientos:
            if "liquidez" in inc.lower():
                condiciones.append(
                    "Mantener razón de liquidez corriente ≥ "
                    f"{politicas.liquidez_minima:.2f}"
                )
            elif "endeudamiento" in inc.lower():
                condiciones.append(
                    "Reducir nivel de endeudamiento a ≤ "
                    f"{politicas.endeudamiento_maximo:.2f}"
                )
            elif "antigüedad" in inc.lower():
                condiciones.append(
                    "Presentar garantías adicionales por antigüedad insuficiente"
                )
            else:
                condiciones.append(f"Regularizar: {inc}")
        return condiciones

    return []


def _generar_recomendaciones(
    estado: EstadoEvaluacion,
    incumplimientos: List[str],
    ratios: RatiosFinancieros,
    info_cred: InfoCrediticia,
) -> List[str]:
    """Genera recomendaciones para el solicitante.

    Args:
        estado: Estado de la evaluación.
        incumplimientos: Lista de incumplimientos.
        ratios: Ratios financieros.
        info_cred: Información crediticia.

    Returns:
        Lista de recomendaciones.
    """
    recomendaciones: List[str] = []

    if ratios.liquidez_corriente < 1.0:
        recomendaciones.append(
            "Mejorar la liquidez corriente aumentando activos circulantes "
            "o reduciendo pasivos de corto plazo"
        )
    if ratios.endeudamiento_total > 0.7:
        recomendaciones.append(
            "Reducir el nivel de endeudamiento total mediante "
            "capitalización de utilidades o aportes de socios"
        )
    if ratios.margen_ebitda < 0.05:
        recomendaciones.append(
            "Mejorar la eficiencia operativa para aumentar el margen EBITDA"
        )
    if info_cred.tiene_protestos:
        recomendaciones.append(
            "Regularizar protestos vigentes ante DICOM"
        )
    if info_cred.tiene_morosidades:
        recomendaciones.append(
            "Regularizar morosidades y mantener historial de pago al día"
        )
    if not recomendaciones:
        recomendaciones.append(
            "Mantener los indicadores financieros actuales"
        )

    return recomendaciones


# ===========================================================================
# Construcción del grafo LangGraph
# ===========================================================================


def construir_grafo() -> StateGraph:
    """Construye y compila el grafo de evaluación crediticia.

    El grafo tiene 4 nodos secuenciales:
    ``ingestion → analisis → evaluacion → dictamen → END``

    Si cualquier nodo produce un error, se transiciona directamente
    al final del grafo.

    Returns:
        ``StateGraph`` compilado listo para ejecutar.
    """
    workflow = StateGraph(GraphState)

    # Registrar nodos
    workflow.add_node("ingestion", nodo_ingestion)
    workflow.add_node("analisis", nodo_analisis)
    workflow.add_node("evaluacion", nodo_evaluacion)
    workflow.add_node("dictamen", nodo_dictamen)

    # Aristas condicionales: si hay error, terminar
    def _ruta_post_ingestion(state: GraphState) -> str:
        if state.get("estado_actual") == "error":
            return "__end__"
        return "analisis"

    def _ruta_post_analisis(state: GraphState) -> str:
        if state.get("estado_actual") == "error":
            return "__end__"
        return "evaluacion"

    def _ruta_post_evaluacion(state: GraphState) -> str:
        if state.get("estado_actual") == "error":
            return "__end__"
        return "dictamen"

    # Conectar nodos
    workflow.set_entry_point("ingestion")
    workflow.add_conditional_edges(
        "ingestion", _ruta_post_ingestion,
        {"analisis": "analisis", "__end__": END},
    )
    workflow.add_conditional_edges(
        "analisis", _ruta_post_analisis,
        {"evaluacion": "evaluacion", "__end__": END},
    )
    workflow.add_conditional_edges(
        "evaluacion", _ruta_post_evaluacion,
        {"dictamen": "dictamen", "__end__": END},
    )
    workflow.add_edge("dictamen", END)

    return workflow.compile()


# ===========================================================================
# Instancia global del grafo (singleton perezoso)
# ===========================================================================

_grafo: Any = None


def get_grafo() -> Any:
    """Retorna la instancia única del grafo compilado.

    El grafo se construye una sola vez (lazy singleton) y se reutiliza
    en todas las invocaciones.

    Returns:
        Grafo LangGraph compilado.
    """
    global _grafo
    if _grafo is None:
        _grafo = construir_grafo()
    return _grafo


# ===========================================================================
# Función de entrada principal
# ===========================================================================


def ejecutar_evaluacion(
    pdf_path: str,
    producto: TipoProducto = TipoProducto.FACTORING,
    politicas: Optional[PoliticasEvaluacion] = None,
) -> ResultadoEvaluacion:
    """Ejecuta el flujo completo de evaluación crediticia.

    Args:
        pdf_path: Ruta al archivo PDF del dossier.
        producto: Tipo de producto (factoring o leasing).
        politicas: Políticas de evaluación personalizadas.
            Si no se proveen, se usan valores por defecto.

    Returns:
        ``ResultadoEvaluacion`` con el dictamen completo.

    Raises:
        RuntimeError: Si el flujo termina en error.
    """
    estado = estado_inicial(pdf_path=pdf_path, producto=producto)
    if politicas is not None:
        estado["politicas_aplicadas"] = politicas
    grafo = get_grafo()
    resultado = grafo.invoke(estado)

    if resultado.get("estado_actual") == "error":
        errores = resultado.get("errores", ["Error desconocido"])
        raise RuntimeError("; ".join(errores))

    resultado_final = resultado.get("resultado")
    if resultado_final is None:
        raise RuntimeError(
            "El flujo terminó sin generar un resultado de evaluación"
        )

    return resultado_final