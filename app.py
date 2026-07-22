"""Aplicación principal Streamlit para evaluación crediticia multiagente.

Conecta la interfaz de usuario con el grafo LangGraph
(``ejecutar_evaluacion`` de ``src.state.graph_state``).
"""

import json
import os
import tempfile
from typing import Optional

import streamlit as st

from src.models.schemas import (
    PoliticasEvaluacion,
    ResultadoEvaluacion,
    TipoProducto,
)
from src.state.graph_state import ejecutar_evaluacion
from src.utils.validators import validar_rut


# --- Configuración de página ---
st.set_page_config(
    page_title="Evaluación Crediticia Chile",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Inicialización de estado de sesión ---
def init_session_state():
    """Inicializa las variables de estado de la sesión de Streamlit."""
    if "resultado" not in st.session_state:
        st.session_state.resultado = None
    if "pdf_procesado" not in st.session_state:
        st.session_state.pdf_procesado = False
    if "error" not in st.session_state:
        st.session_state.error = None
    if "etapa_actual" not in st.session_state:
        st.session_state.etapa_actual = "carga"


init_session_state()


# --- Sidebar: Configuración ---
with st.sidebar:
    st.title("⚙️ Configuración")
    st.markdown("---")

    # Selección de producto
    producto_seleccionado = st.selectbox(
        "Producto Crediticio",
        options=[p.value for p in TipoProducto],
        format_func=lambda x: {
            "factoring": "Factoring",
            "leasing": "Leasing",
        }.get(x, x.capitalize()),
        help="Selecciona el tipo de producto a evaluar",
    )

    st.markdown("### Parámetros de Evaluación")
    antiguedad_min = st.number_input(
        "Antigüedad mínima (meses)",
        min_value=0,
        value=12,
        step=6,
        help="Meses mínimos de operación de la empresa",
    )
    liquidez_min = st.number_input(
        "Liquidez mínima",
        min_value=0.0,
        value=1.2,
        step=0.1,
        format="%.1f",
        help="Razón circulante mínima requerida",
    )
    endeudamiento_max = st.number_input(
        "Endeudamiento máximo",
        min_value=0.0,
        value=2.5,
        step=0.1,
        format="%.1f",
        help="Endeudamiento máximo permitido",
    )
    cobertura_ebitda_min = st.number_input(
        "Cobertura EBITDA mínima (leasing)",
        min_value=0.0,
        value=1.5,
        step=0.1,
        format="%.1f",
        help="Aplica solo para evaluación de leasing",
    )

    st.markdown("---")
    st.markdown(
        """
        **Powered by:**
        - LangChain / LangGraph
        - Groq API (Llama 3)
        - PyPDF / Pandas
        """
    )


# --- Cabecera principal ---
st.title("📊 Sistema de Evaluación Crediticia")
st.markdown(
    """
    Sistema multiagente para evaluación de **factoring** y **leasing** en Chile.
    Procesa carpetas tributarias y financieras para generar dictámenes de riesgo
    basados en políticas CMF y parámetros internos configurables.
    """
)

st.markdown("---")


# --- Sección de carga de PDF ---
st.header("📄 Carga de Documentos")

col1, col2 = st.columns([3, 1])

with col1:
    archivo_subido = st.file_uploader(
        "Selecciona el dossier PDF de la empresa",
        type=["pdf"],
        help="Sube el PDF con la carpeta tributaria (F29, F22, Balance, DICOM)",
    )

with col2:
    st.markdown("### Estado")
    if st.session_state.pdf_procesado:
        st.success("✅ PDF cargado")
    else:
        st.info("⏳ Pendiente")


# --- Botón de evaluación ---
if archivo_subido is not None:
    st.session_state.pdf_procesado = True
    st.session_state.archivo = archivo_subido

    if st.button("🚀 Ejecutar Evaluación", type="primary", use_container_width=True):
        # ── Indicador visual de etapas ──────────────────────────────
        status_placeholder = st.empty()

        with status_placeholder:
            etapas = st.status(
                "Ejecutando evaluación crediticia…",
                expanded=True,
                state="running",
            )

            with etapas:
                st.write("Flujo del grafo LangGraph:")

                paso_ingestion = st.empty()
                paso_analisis = st.empty()
                paso_evaluacion = st.empty()
                paso_dictamen = st.empty()

                paso_ingestion.write("⬜ 1/4  Ingestion — extrayendo datos del PDF…")

        try:
            # Guardar PDF temporal
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(archivo_subido.getvalue())
                pdf_path = tmp.name

            with status_placeholder:
                with etapas:
                    paso_ingestion.write("🔄 1/4  Ingestion — extrayendo datos del PDF…")

            # ── Ejecutar el grafo real ──────────────────────────────
            politicas = PoliticasEvaluacion(
                producto=TipoProducto(producto_seleccionado),
                antiguedad_minima_meses=antiguedad_min,
                liquidez_minima=liquidez_min,
                endeudamiento_maximo=endeudamiento_max,
                cobertura_ebitda_minima=cobertura_ebitda_min,
            )

            with status_placeholder:
                with etapas:
                    paso_ingestion.write("✅ 1/4  Ingestion — datos extraídos")
                    paso_analisis.write("🔄 2/4  Análisis — calculando ratios financieros…")

            resultado = ejecutar_evaluacion(
                pdf_path=pdf_path,
                producto=TipoProducto(producto_seleccionado),
                politicas=politicas,
            )

            with status_placeholder:
                with etapas:
                    paso_ingestion.write("✅ 1/4  Ingestion — datos extraídos")
                    paso_analisis.write("✅ 2/4  Análisis — ratios calculados")
                    paso_evaluacion.write("✅ 3/4  Evaluación de riesgo — políticas aplicadas")
                    paso_dictamen.write("✅ 4/4  Dictamen — generado")
                    etapas.update(
                        label="Evaluación completada exitosamente",
                        state="complete",
                    )

            st.session_state.resultado = resultado
            st.session_state.etapa_actual = "resultados"

            # Limpiar archivo temporal
            os.unlink(pdf_path)

        except Exception as e:
            with status_placeholder:
                with etapas:
                    etapas.update(
                        label=f"Error: {e}",
                        state="error",
                    )
            st.session_state.error = str(e)
            st.session_state.etapa_actual = "error"

    # Limpiar si se sube nuevo archivo
    if st.button("🔄 Nueva Evaluación"):
        st.session_state.resultado = None
        st.session_state.pdf_procesado = False
        st.session_state.error = None
        st.session_state.etapa_actual = "carga"
        st.rerun()


# --- Visualización de resultados ---
if st.session_state.resultado is not None:
    resultado = st.session_state.resultado

    st.markdown("---")
    st.header("📋 Resultados de la Evaluación")

    # Resumen ejecutivo
    col_estado, col_puntaje, col_producto = st.columns(3)
    with col_estado:
        estado_color = {
            "aprobado": "🟢",
            "aprobado_con_condiciones": "🟡",
            "rechazado": "🔴",
            "requiere_analisis_manual": "🟠",
        }
        emoji = estado_color.get(resultado.estado.value, "⚪")
        st.metric(
            "Estado",
            f"{emoji} {resultado.estado.value.replace('_', ' ').title()}",
        )

    with col_puntaje:
        st.metric("Puntaje de Riesgo", f"{resultado.puntaje_riesgo:.0f}/100")

    with col_producto:
        st.metric("Producto", resultado.producto.value.title())

    # Ratios financieros
    st.markdown("### 📈 Ratios Financieros")
    r = resultado.ratios_calculados

    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    with col_r1:
        st.metric("Liquidez Corriente", f"{r.liquidez_corriente:.2f}")
    with col_r2:
        st.metric("Endeudamiento Total", f"{r.endeudamiento_total:.2f}")
    with col_r3:
        st.metric("Margen EBITDA", f"{r.margen_ebitda:.2%}")
    with col_r4:
        st.metric("ROI", f"{r.roi:.2%}")

    if resultado.producto == TipoProducto.FACTORING:
        st.metric("Rotación de Cartera", f"{r.rotacion_cartera:.2f}")
    else:
        st.metric(
            "Cobertura Servicio Deuda FCF",
            f"{r.cobertura_servicio_deuda_fcf:.2f}",
        )

    # Factores
    col_f1, col_f2 = st.columns(2)

    with col_f1:
        st.markdown("### ✅ Factores Positivos")
        if resultado.factores_positivos:
            for factor in resultado.factores_positivos:
                st.success(f"✓ {factor}")
        else:
            st.info("Sin factores positivos identificados")

    with col_f2:
        st.markdown("### ⚠️ Factores de Riesgo")
        if resultado.factores_riesgo:
            for riesgo in resultado.factores_riesgo:
                st.error(f"✗ {riesgo}")
        else:
            st.info("Sin factores de riesgo identificados")

    # Dictamen completo
    with st.expander("📝 Ver Dictamen Completo", expanded=True):
        st.text_area(
            "Dictamen Crediticio",
            value=resultado.dictamen,
            height=400,
            disabled=True,
        )

    # Condiciones y recomendaciones
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.markdown("### 📋 Condiciones Aplicables")
        if resultado.condiciones_aplicables:
            for condicion in resultado.condiciones_aplicables:
                st.warning(f"📌 {condicion}")
        else:
            st.success("Sin condiciones especiales")

    with col_c2:
        st.markdown("### 💡 Recomendaciones")
        if resultado.recomendaciones:
            for rec in resultado.recomendaciones:
                st.info(f"→ {rec}")

    # Descargar resultados
    st.markdown("### 📥 Descargar Resultados")

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        dictamen_texto = (
            f"DICTAMEN CREDITICIO\n"
            f"{'='*50}\n\n"
            f"{resultado.dictamen}\n\n"
            f"FACTORES POSITIVOS:\n"
            f"{chr(10).join('- ' + f for f in resultado.factores_positivos)}\n\n"
            f"FACTORES DE RIESGO:\n"
            f"{chr(10).join('- ' + f for f in resultado.factores_riesgo)}\n\n"
            f"CONDICIONES:\n"
            f"{chr(10).join('- ' + c for c in resultado.condiciones_aplicables)}\n\n"
            f"RECOMENDACIONES:\n"
            f"{chr(10).join('- ' + r for r in resultado.recomendaciones)}"
        )
        st.download_button(
            label="📄 Descargar Dictamen (TXT)",
            data=dictamen_texto,
            file_name=f"dictamen_{resultado.rut_empresa}_{resultado.producto.value}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with col_d2:
        resultado_json = resultado.model_dump_json(indent=2)
        st.download_button(
            label="📊 Descargar JSON",
            data=resultado_json,
            file_name=f"evaluacion_{resultado.rut_empresa}_{resultado.producto.value}.json",
            mime="application/json",
            use_container_width=True,
        )


# --- Manejo de errores ---
if st.session_state.error:
    st.error(f"❌ Error en la evaluación: {st.session_state.error}")
    if st.button("Reintentar"):
        st.session_state.error = None
        st.session_state.resultado = None
        st.session_state.pdf_procesado = False
        st.session_state.etapa_actual = "carga"
        st.rerun()


# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
        <p>Sistema de Evaluación Crediticia Multiagente | Chile</p>
        <p>Powered by LangChain, LangGraph y Groq API</p>
    </div>
    """,
    unsafe_allow_html=True,
)