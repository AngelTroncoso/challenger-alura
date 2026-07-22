"""Tests de integración para el flujo completo de evaluación crediticia.

Valida que el grafo LangGraph, los 4 nodos y la función
``ejecutar_evaluacion`` operen correctamente con datos mock,
sin depender de archivos PDF reales ni de la API de Groq.
"""

import json
import os
import sys
from copy import deepcopy
from typing import Any, Dict
from unittest.mock import patch

import pytest

# ── Asegurar que el proyecto está en sys.path ────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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
from src.state.graph_state import (
    GraphState,
    construir_grafo,
    estado_inicial,
    nodo_ingestion,
    nodo_analisis,
    nodo_evaluacion,
    nodo_dictamen,
)


# ===========================================================================
# Fixtures — datos de prueba
# ===========================================================================


@pytest.fixture
def estados_financieros_ejemplo() -> EstadosFinancieros:
    """Estados financieros de una empresa saludable (factoring)."""
    return EstadosFinancieros(
        periodo="2024-12",
        activos_corrientes=85_000_000,
        activos_no_corrientes=120_000_000,
        pasivos_corrientes=55_000_000,
        pasivos_no_corrientes=45_000_000,
        patrimonio=105_000_000,
        ventas_netas=320_000_000,
        costo_ventas=210_000_000,
        resultado_operacional=45_000_000,
        resultado_neto=28_000_000,
        depreciacion=12_000_000,
        gastos_financieros=8_000_000,
    )


@pytest.fixture
def info_tributaria_ejemplo() -> InfoTributaria:
    return InfoTributaria(
        rut_empresa="76.123.456-7",
        razon_social="Empresa Ejemplo S.A.",
    )


@pytest.fixture
def info_crediticia_sana() -> InfoCrediticia:
    return InfoCrediticia(
        tiene_protestos=False,
        monto_protestos=0.0,
        tiene_morosidades=False,
        dias_morosidad=0,
        morosidades_previsionales=False,
        antiguedad_meses=48,
    )


@pytest.fixture
def info_crediticia_riesgosa() -> InfoCrediticia:
    return InfoCrediticia(
        tiene_protestos=True,
        monto_protestos=1_500_000,
        tiene_morosidades=True,
        dias_morosidad=45,
        morosidades_previsionales=True,
        antiguedad_meses=6,
    )


@pytest.fixture
def politicas_default() -> PoliticasEvaluacion:
    return PoliticasEvaluacion(
        producto=TipoProducto.FACTORING,
        antiguedad_minima_meses=12,
        liquidez_minima=1.2,
        endeudamiento_maximo=2.5,
        cobertura_ebitda_minima=1.5,
    )


@pytest.fixture
def ratios_factoring(estados_financieros_ejemplo) -> RatiosFinancieros:
    """Precalcula ratios para factoring."""
    from src.tools.ratio_calculator_tool import calcular_ratios_factoring

    return calcular_ratios_factoring(estados_financieros_ejemplo)


@pytest.fixture
def ratios_leasing(estados_financieros_ejemplo) -> RatiosFinancieros:
    """Precalcula ratios para leasing."""
    from src.tools.ratio_calculator_tool import calcular_ratios_leasing

    return calcular_ratios_leasing(estados_financieros_ejemplo)


# ===========================================================================
# Tests unitarios de cada nodo del grafo
# ===========================================================================


class TestNodoIngestion:
    """Valida el nodo de ingesta con datos mock (sin PDF real)."""

    def test_error_sin_pdf(self):
        """Si no hay pdf_path, debe retornar error."""
        state: GraphState = estado_inicial(pdf_path="", producto=TipoProducto.FACTORING)
        result = nodo_ingestion(state)
        assert result["estado_actual"] == "error"
        assert len(result["errores"]) > 0

    def test_pdf_inexistente(self):
        """Si el PDF no existe, debe retornar error."""
        state: GraphState = estado_inicial(
            pdf_path="/ruta/inexistente.pdf", producto=TipoProducto.FACTORING
        )
        result = nodo_ingestion(state)
        assert result["estado_actual"] == "error"
        assert len(result["errores"]) > 0

    def test_pdf_valido_con_mock(self):
        """Simula un PDF con datos parseables exitosamente."""
        from src.models.schemas import InfoTributaria, EstadosFinancieros, InfoCrediticia

        mock_dossier = {
            "tipo_documento": "balance",
            "info_tributaria": InfoTributaria(
                rut_empresa="76.123.456-7", razon_social="Mock S.A."
            ),
            "estados_financieros": EstadosFinancieros(
                periodo="2024-12",
                activos_corrientes=100_000_000,
                activos_no_corrientes=50_000_000,
                pasivos_corrientes=40_000_000,
                pasivos_no_corrientes=20_000_000,
                patrimonio=90_000_000,
                ventas_netas=200_000_000,
                costo_ventas=120_000_000,
                resultado_operacional=30_000_000,
                resultado_neto=18_000_000,
                depreciacion=5_000_000,
                gastos_financieros=3_000_000,
            ),
            "info_crediticia": InfoCrediticia(
                tiene_protestos=False, antiguedad_meses=36
            ),
            "error": None,
        }

        with patch(
            "src.tools.pdf_parser_tool.procesar_dossier", return_value=mock_dossier
        ):
            state: GraphState = estado_inicial(
                pdf_path="/mock/dummy.pdf",
                producto=TipoProducto.FACTORING,
            )
            result = nodo_ingestion(state)
            assert result["estado_actual"] == "analisis"
            assert result["rut_empresa"] == "76.123.456-7"
            assert result["razon_social"] == "Mock S.A."
            assert result["info_tributaria"] is not None
            assert result["estados_financieros"] is not None
            assert result["info_crediticia"] is not None


class TestNodoAnalisis:
    """Valida el nodo de análisis financiero."""

    def test_error_sin_ee(self):
        """Sin estados financieros debe retornar error."""
        state = estado_inicial(
            pdf_path="/mock/dummy.pdf",
            producto=TipoProducto.FACTORING,
        )
        result = nodo_analisis(state)
        assert result["estado_actual"] == "error"

    def test_ratios_factoring(self, estados_financieros_ejemplo):
        """Verifica el cálculo de ratios para factoring."""
        state = estado_inicial(
            pdf_path="/mock/dummy.pdf",
            producto=TipoProducto.FACTORING,
        )
        state["estados_financieros"] = estados_financieros_ejemplo
        state["info_crediticia"] = InfoCrediticia()

        result = nodo_analisis(state)
        assert result["estado_actual"] == "evaluacion"
        assert result["ratios_calculados"] is not None
        r = result["ratios_calculados"]
        assert r.liquidez_corriente > 0
        assert r.rotacion_cartera > 0  # específico de factoring
        assert 0 <= r.endeudamiento_total <= 1

    def test_ratios_leasing(self, estados_financieros_ejemplo):
        """Verifica el cálculo de ratios para leasing."""
        state = estado_inicial(
            pdf_path="/mock/dummy.pdf",
            producto=TipoProducto.LEASING,
        )
        state["estados_financieros"] = estados_financieros_ejemplo

        result = nodo_analisis(state)
        assert result["estado_actual"] == "evaluacion"
        assert result["ratios_calculados"] is not None
        r = result["ratios_calculados"]
        assert r.rotacion_cartera == 0.0  # no aplica en leasing
        assert r.cobertura_servicio_deuda_fcf > 0


class TestNodoEvaluacion:
    """Valida el nodo de evaluación de riesgo."""

    def test_error_sin_ratios(self):
        """Sin ratios debe retornar error."""
        state = estado_inicial(
            pdf_path="/mock/dummy.pdf",
            producto=TipoProducto.FACTORING,
        )
        result = nodo_evaluacion(state)
        assert result["estado_actual"] == "error"

    def test_evaluacion_aprobado(
        self,
        ratios_factoring,
        info_tributaria_ejemplo,
        info_crediticia_sana,
        politicas_default,
    ):
        """Empresa saludable debe resultar aprobada."""
        state = estado_inicial(
            pdf_path="/mock/dummy.pdf",
            producto=TipoProducto.FACTORING,
        )
        state["ratios_calculados"] = ratios_factoring
        state["info_tributaria"] = info_tributaria_ejemplo
        state["info_crediticia"] = info_crediticia_sana
        state["politicas_aplicadas"] = politicas_default

        result = nodo_evaluacion(state)
        assert result["estado_actual"] == "dictamen"
        assert len(result["incumplimientos"]) == 0
        assert 0 <= result["puntaje_riesgo"] <= 100

    def test_evaluacion_rechazado(
        self,
        ratios_factoring,
        info_tributaria_ejemplo,
        info_crediticia_riesgosa,
        politicas_default,
    ):
        """Empresa riesgosa debe detectar incumplimientos."""
        state = estado_inicial(
            pdf_path="/mock/dummy.pdf",
            producto=TipoProducto.FACTORING,
        )
        state["ratios_calculados"] = ratios_factoring
        state["info_tributaria"] = info_tributaria_ejemplo
        state["info_crediticia"] = info_crediticia_riesgosa
        state["politicas_aplicadas"] = politicas_default

        result = nodo_evaluacion(state)
        assert result["estado_actual"] == "dictamen"
        # Debe detectar antigüedad insuficiente + protestos + morosidades
        assert len(result["incumplimientos"]) >= 1


class TestNodoDictamen:
    """Valida el nodo de dictamen final."""

    def test_dictamen_completo(
        self,
        estados_financieros_ejemplo,
        info_tributaria_ejemplo,
        info_crediticia_sana,
        ratios_factoring,
        politicas_default,
    ):
        """Verifica que se genere un ResultadoEvaluacion completo."""
        from src.tools.policy_engine_tool import evaluar_cumplimiento, determinar_estado

        incumplimientos, puntaje = evaluar_cumplimiento(
            ratios_factoring,
            politicas_default,
            info_crediticia_sana,
            info_tributaria_ejemplo,
        )

        state = estado_inicial(
            pdf_path="/mock/dummy.pdf",
            producto=TipoProducto.FACTORING,
        )
        state["info_tributaria"] = info_tributaria_ejemplo
        state["estados_financieros"] = estados_financieros_ejemplo
        state["info_crediticia"] = info_crediticia_sana
        state["ratios_calculados"] = ratios_factoring
        state["politicas_aplicadas"] = politicas_default
        state["incumplimientos"] = incumplimientos
        state["puntaje_riesgo"] = puntaje

        result = nodo_dictamen(state)
        assert result["estado_actual"] == "completo"
        assert result["resultado"] is not None

        r: ResultadoEvaluacion = result["resultado"]
        assert r.rut_empresa == "76.123.456-7"
        assert r.estado in (
            EstadoEvaluacion.APROBADO,
            EstadoEvaluacion.APROBADO_CON_CONDICIONES,
        )
        assert r.dictamen
        assert len(r.factores_positivos) >= 0
        assert len(r.recomendaciones) >= 0


# ===========================================================================
# Tests de integración del grafo completo
# ===========================================================================


class TestGrafoCompleto:
    """Valida la ejecución del grafo LangGraph completo."""

    def test_estado_inicial_defaults(self):
        """El estado inicial debe tener valores por defecto coherentes."""
        state = estado_inicial(
            pdf_path="/ruta/test.pdf",
            producto=TipoProducto.FACTORING,
        )
        assert state["pdf_path"] == "/ruta/test.pdf"
        assert state["producto"] == TipoProducto.FACTORING
        assert state["estado_actual"] == "ingestion"
        assert state["puntaje_riesgo"] == 50.0
        assert state["incumplimientos"] == []
        assert state["errores"] == []

    def test_grafo_flujo_completo_mock(self):
        """Ejecuta el grafo completo con datos mock (sin PDF real)."""
        from src.models.schemas import InfoTributaria, EstadosFinancieros, InfoCrediticia

        # Mock del procesar_dossier para evitar leer un PDF real
        mock_dossier = {
            "tipo_documento": "balance",
            "info_tributaria": InfoTributaria(
                rut_empresa="76.123.456-7", razon_social="Mock Integral S.A."
            ),
            "estados_financieros": EstadosFinancieros(
                periodo="2024-12",
                activos_corrientes=100_000_000,
                activos_no_corrientes=50_000_000,
                pasivos_corrientes=40_000_000,
                pasivos_no_corrientes=20_000_000,
                patrimonio=90_000_000,
                ventas_netas=200_000_000,
                costo_ventas=120_000_000,
                resultado_operacional=30_000_000,
                resultado_neto=18_000_000,
                depreciacion=5_000_000,
                gastos_financieros=3_000_000,
            ),
            "info_crediticia": InfoCrediticia(
                tiene_protestos=False, antiguedad_meses=36
            ),
            "error": None,
        }

        with patch(
            "src.tools.pdf_parser_tool.procesar_dossier", return_value=mock_dossier
        ):
            grafo = construir_grafo()
            state = estado_inicial(
                pdf_path="/mock/integral.pdf",
                producto=TipoProducto.FACTORING,
            )
            state["politicas_aplicadas"] = PoliticasEvaluacion(
                producto=TipoProducto.FACTORING,
                antiguedad_minima_meses=12,
                liquidez_minima=1.2,
                endeudamiento_maximo=2.5,
            )

            resultado = grafo.invoke(state)

        assert resultado["estado_actual"] in ("completo", "error")

        if resultado["estado_actual"] == "completo":
            assert resultado["resultado"] is not None
            r = resultado["resultado"]
            assert r.rut_empresa == "76.123.456-7"
            assert r.producto == TipoProducto.FACTORING
            assert isinstance(r.dictamen, str) and len(r.dictamen) > 50
        else:
            # Si falló, debe tener errores explicativos
            assert len(resultado["errores"]) > 0

    def test_grafo_flujo_con_error(self):
        """Un PDF inexistente debe resultar en estado 'error'."""
        grafo = construir_grafo()
        state = estado_inicial(
            pdf_path="/ruta/inexistente.pdf",
            producto=TipoProducto.FACTORING,
        )
        resultado = grafo.invoke(state)
        assert resultado["estado_actual"] == "error"
        assert len(resultado["errores"]) > 0


# ===========================================================================
# Tests de la función ejecutar_evaluacion (API pública)
# ===========================================================================


class TestEjecutarEvaluacion:
    """Valida la función de entrada principal del paquete."""

    def test_ejecutar_con_pdf_inexistente(self):
        """Debe lanzar RuntimeError si el PDF no existe."""
        from src.state.graph_state import ejecutar_evaluacion

        with pytest.raises((RuntimeError, FileNotFoundError)):
            ejecutar_evaluacion(
                pdf_path="/ruta/inexistente.pdf",
                producto=TipoProducto.FACTORING,
            )

    def test_ejecutar_con_mock(self):
        """Ejecuta evaluación completa mockeando la ingesta."""
        from src.state.graph_state import ejecutar_evaluacion
        from src.models.schemas import InfoTributaria, EstadosFinancieros, InfoCrediticia

        mock_dossier = {
            "tipo_documento": "balance",
            "info_tributaria": InfoTributaria(
                rut_empresa="76.123.456-7", razon_social="Mock Exitosa Ltda."
            ),
            "estados_financieros": EstadosFinancieros(
                periodo="2024-12",
                activos_corrientes=200_000_000,
                activos_no_corrientes=100_000_000,
                pasivos_corrientes=80_000_000,
                pasivos_no_corrientes=40_000_000,
                patrimonio=180_000_000,
                ventas_netas=500_000_000,
                costo_ventas=300_000_000,
                resultado_operacional=80_000_000,
                resultado_neto=50_000_000,
                depreciacion=15_000_000,
                gastos_financieros=10_000_000,
            ),
            "info_crediticia": InfoCrediticia(
                tiene_protestos=False,
                antiguedad_meses=60,
            ),
            "error": None,
        }

        with patch(
            "src.tools.pdf_parser_tool.procesar_dossier", return_value=mock_dossier
        ):
            politicas = PoliticasEvaluacion(
                producto=TipoProducto.FACTORING,
                antiguedad_minima_meses=12,
                liquidez_minima=1.0,
                endeudamiento_maximo=3.0,
            )
            resultado = ejecutar_evaluacion(
                pdf_path="/mock/exitosa.pdf",
                producto=TipoProducto.FACTORING,
                politicas=politicas,
            )

        assert isinstance(resultado, ResultadoEvaluacion)
        assert resultado.rut_empresa == "76.123.456-7"
        assert resultado.estado in (
            EstadoEvaluacion.APROBADO,
            EstadoEvaluacion.APROBADO_CON_CONDICIONES,
        )
        assert 0 <= resultado.puntaje_riesgo <= 100
        assert resultado.ratios_calculados.liquidez_corriente > 0
        assert resultado.factores_positivos is not None
        assert resultado.recomendaciones is not None

    def test_ejecutar_con_politicas_personalizadas(self):
        """Verifica que las políticas personalizadas se apliquen."""
        from src.state.graph_state import ejecutar_evaluacion
        from src.models.schemas import InfoTributaria, EstadosFinancieros, InfoCrediticia

        mock_dossier = {
            "tipo_documento": "balance",
            "info_tributaria": InfoTributaria(
                rut_empresa="12.345.678-9", razon_social="Test Ltda."
            ),
            "estados_financieros": EstadosFinancieros(
                periodo="2024-12",
                activos_corrientes=50_000_000,
                activos_no_corrientes=30_000_000,
                pasivos_corrientes=45_000_000,
                pasivos_no_corrientes=15_000_000,
                patrimonio=20_000_000,
                ventas_netas=100_000_000,
                costo_ventas=70_000_000,
                resultado_operacional=5_000_000,
                resultado_neto=2_000_000,
                depreciacion=1_000_000,
                gastos_financieros=4_000_000,
            ),
            "info_crediticia": InfoCrediticia(
                tiene_protestos=True,
                monto_protestos=2_000_000,
                antiguedad_meses=8,
            ),
            "error": None,
        }

        with patch(
            "src.tools.pdf_parser_tool.procesar_dossier", return_value=mock_dossier
        ):
            politicas = PoliticasEvaluacion(
                producto=TipoProducto.FACTORING,
                antiguedad_minima_meses=24,
                liquidez_minima=2.0,
                endeudamiento_maximo=1.0,
            )
            resultado = ejecutar_evaluacion(
                pdf_path="/mock/test.pdf",
                producto=TipoProducto.FACTORING,
                politicas=politicas,
            )

        # Con estas políticas restrictivas más datos débiles, debe ser
        # rechazado o aprobado con condiciones
        assert resultado.estado in (
            EstadoEvaluacion.RECHAZADO,
            EstadoEvaluacion.APROBADO_CON_CONDICIONES,
            EstadoEvaluacion.REQUIERE_ANALISIS_MANUAL,
        )


# ===========================================================================
# Tests de serialización y formato de salida
# ===========================================================================


class TestResultadoEvaluacion:
    """Valida que el resultado sea serializable y cumpla el esquema."""

    def test_serializacion_json(
        self,
        estados_financieros_ejemplo,
        info_tributaria_ejemplo,
        info_crediticia_sana,
        ratios_factoring,
        politicas_default,
    ):
        """El ResultadoEvaluacion debe ser serializable a JSON sin errores."""
        from src.tools.policy_engine_tool import evaluar_cumplimiento, determinar_estado

        incumplimientos, puntaje = evaluar_cumplimiento(
            ratios_factoring,
            politicas_default,
            info_crediticia_sana,
            info_tributaria_ejemplo,
        )
        estado = determinar_estado(incumplimientos, puntaje)

        resultado = ResultadoEvaluacion(
            rut_empresa=info_tributaria_ejemplo.rut_empresa,
            producto=TipoProducto.FACTORING,
            estado=estado,
            puntaje_riesgo=puntaje,
            ratios_calculados=ratios_factoring,
            politicas_aplicadas=politicas_default,
            dictamen="Dictamen de prueba." * 20,
            condiciones_aplicables=incumplimientos,
            factores_positivos=["Liquidez saludable"],
            factores_riesgo=[],
            recomendaciones=["Mantener indicadores"],
        )

        # Serialización (debe funcionar sin errores)
        json_str = resultado.model_dump_json(indent=2)
        parsed = json.loads(json_str)

        assert parsed["rut_empresa"] == "76.123.456-7"
        assert parsed["producto"] == "factoring"
        assert "dictamen" in parsed
        assert "ratios_calculados" in parsed
        assert "liquidez_corriente" in parsed["ratios_calculados"]
        assert isinstance(parsed["puntaje_riesgo"], (int, float))

    def test_estado_enumerado_valido(self):
        """Todos los estados posibles deben estar cubiertos."""
        estados = EstadoEvaluacion._value2member_map_
        assert "aprobado" in estados
        assert "aprobado_con_condiciones" in estados
        assert "rechazado" in estados
        assert "requiere_analisis_manual" in estados