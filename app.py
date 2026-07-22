"""Aplicación principal Streamlit para evaluación crediticia multiagente.

Conecta la interfaz de usuario con el grafo LangGraph
(``ejecutar_evaluacion`` de ``src.state.graph_state``).
"""

import json
import os
import tempfile
from typing import Optional, List

import streamlit as st

from src.models.schemas import (
    PoliticasEvaluacion,
    ResultadoEvaluacion,
    TipoProducto,
)
from src.state.graph_state import ejecutar_evaluacion
from src.utils.validators import validar_rut


# ──────────────────────────────────────────────────────────────────────
#  CONSTANTES
# ──────────────────────────────────────────────────────────────────────
DOCUMENTOS_REQUERIDOS = [
    {
        "id": "carpeta_tributaria",
        "nombre": "Carpeta Tributaria CMF/SII",
        "icono": "📁",
        "descripcion": "Resumen de obligaciones tributarias emitido por CMF o SII. "
                       "Permite verificar el cumplimiento fiscal y detectar morosidades.",
        "palabras_clave": ["carpeta", "tributaria", "cmf", "sii", "tributario"],
    },
    {
        "id": "f29",
        "nombre": "Formulario 29 (F29 - Últimos 12 meses)",
        "icono": "📋",
        "descripcion": "Declaración mensual de IVA y otros impuestos. "
                       "Fundamental para estimar ingresos recurrentes y estacionalidad.",
        "palabras_clave": ["f29", "formulario 29", "declaracion", "iva", "dii"],
    },
    {
        "id": "balance",
        "nombre": "Balance General y Estado de Resultados",
        "icono": "📊",
        "descripcion": "Estados financieros que reflejan la situación patrimonial y "
                       "rendimiento de la empresa. Base para calcular ratios de liquidez, "
                       "endeudamiento y rentabilidad.",
        "palabras_clave": ["balance", "estado resultado", "eeff", "financiero", "pérdidas", "ganancias"],
    },
    {
        "id": "cartola",
        "nombre": "Cartola Bancaria reciente",
        "icono": "🏦",
        "descripcion": "Movimientos bancarios de los últimos 3-6 meses. "
                       "Permite validar flujo de caja real, concentración de ingresos "
                       "y comportamiento de pagos.",
        "palabras_clave": ["cartola", "bancaria", "banco", "movimiento", "estado cuenta"],
    },
    {
        "id": "estatutos",
        "nombre": "Certificado de Estatutos/Vigencia",
        "icono": "📜",
        "descripcion": "Documento que acredita la existencia legal y vigencia de la "
                       "sociedad. Necesario para verificar representación legal y "
                       "facultades para contratar.",
        "palabras_clave": ["estatuto", "vigencia", "constitucion", "sociedad", "escritura"],
    },
]

NIVELES_COBERTURA = [
    {"min": 100, "max": 100, "label": "Completo", "precision": "Alta (>95%)", "color": "#00CC66"},
    {"min": 80, "max": 99, "label": "Casi completo", "precision": "Alta (85-95%)", "color": "#66CC00"},
    {"min": 60, "max": 79, "label": "Parcial", "precision": "Media (70-85%)", "color": "#FFAA00"},
    {"min": 40, "max": 59, "label": "Bajo", "precision": "Media-Baja (50-70%)", "color": "#FF6600"},
    {"min": 0, "max": 39, "label": "Insuficiente", "precision": "Baja (<50%)", "color": "#FF3333"},
]


# ──────────────────────────────────────────────────────────────────────
#  FUNCIONES AUXILIARES
# ──────────────────────────────────────────────────────────────────────
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
    if "archivos_subidos" not in st.session_state:
        st.session_state.archivos_subidos = []
    if "cobertura_documental" not in st.session_state:
        st.session_state.cobertura_documental = {}


def identificar_documento(nombre_archivo: str) -> Optional[str]:
    """Intenta identificar a qué documento requerido corresponde un archivo por su nombre."""
    nombre_lower = nombre_archivo.lower().replace("_", " ").replace("-", " ")
    for doc in DOCUMENTOS_REQUERIDOS:
        for kw in doc["palabras_clave"]:
            if kw in nombre_lower:
                return doc["id"]
    return None


def analizar_cobertura(archivos: List) -> dict:
    """Analiza qué documentos han sido cubiertos por los archivos subidos."""
    cobertura = {doc["id"]: False for doc in DOCUMENTOS_REQUERIDOS}
    for archivo in archivos:
        doc_id = identificar_documento(archivo.name)
        if doc_id:
            cobertura[doc_id] = True
    return cobertura


def calcular_porcentaje(cobertura: dict) -> int:
    """Calcula el porcentaje de completitud del dossier."""
    if not cobertura:
        return 0
    cubiertos = sum(1 for v in cobertura.values() if v)
    return int((cubiertos / len(cobertura)) * 100)


def obtener_nivel_cobertura(porcentaje: int) -> dict:
    """Obtiene el nivel de cobertura según el porcentaje."""
    for nivel in NIVELES_COBERTURA:
        if nivel["min"] <= porcentaje <= nivel["max"]:
            return nivel
    return NIVELES_COBERTURA[-1]


def render_dark_mode_css():
    """Renderiza CSS personalizado para modo oscuro."""
    st.markdown(
        """
        <style>
        /* ── Fondo general oscuro ── */
        .stApp {
            background-color: #0E1117;
            color: #E0E0E0;
        }

        /* ── Tarjetas / contenedores ── */
        .doc-card {
            background: linear-gradient(135deg, #1A1D27 0%, #22263A 100%);
            border: 1px solid #2D3142;
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 10px;
            transition: all 0.2s ease;
        }
        .doc-card:hover {
            border-color: #4A4F6A;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            transform: translateY(-1px);
        }

        /* ── Badge de estado ── */
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: 600;
            white-space: nowrap;
        }
        .status-badge.detected {
            background: rgba(0, 204, 102, 0.15);
            color: #00CC66;
            border: 1px solid rgba(0, 204, 102, 0.3);
        }
        .status-badge.pending {
            background: rgba(255, 170, 0, 0.15);
            color: #FFAA00;
            border: 1px solid rgba(255, 170, 0, 0.3);
        }
        .status-badge.missing {
            background: rgba(255, 51, 51, 0.15);
            color: #FF3333;
            border: 1px solid rgba(255, 51, 51, 0.3);
        }

        /* ── Tooltip personalizado ── */
        .tooltip-container {
            position: relative;
            display: inline-flex;
            align-items: center;
            cursor: help;
        }
        .tooltip-container .tooltip-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #2D3142;
            color: #8A8FA8;
            font-size: 12px;
            font-weight: bold;
            margin-left: 8px;
            transition: all 0.2s;
        }
        .tooltip-container:hover .tooltip-icon {
            background: #4A4F6A;
            color: #E0E0E0;
        }
        .tooltip-container .tooltip-text {
            visibility: hidden;
            opacity: 0;
            width: 280px;
            background: #1A1D27;
            border: 1px solid #4A4F6A;
            border-radius: 8px;
            padding: 12px 14px;
            position: absolute;
            z-index: 1000;
            bottom: calc(100% + 8px);
            left: 50%;
            transform: translateX(-50%);
            transition: opacity 0.2s, visibility 0.2s;
            font-size: 0.85em;
            line-height: 1.5;
            color: #C0C4D0;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5);
            pointer-events: none;
        }
        .tooltip-container:hover .tooltip-text {
            visibility: visible;
            opacity: 1;
        }

        /* ── Tarjeta de cobertura ── */
        .coverage-card {
            background: linear-gradient(135deg, #1A1D27 0%, #1F2340 100%);
            border: 1px solid #2D3142;
            border-radius: 16px;
            padding: 24px;
            margin: 16px 0;
        }
        .coverage-card h3 {
            margin: 0 0 16px 0;
            color: #E0E0E0;
            font-size: 1.1em;
        }

        /* ── Barra de progreso personalizada ── */
        .progress-bar-container {
            width: 100%;
            height: 12px;
            background: #2D3142;
            border-radius: 6px;
            overflow: hidden;
            margin: 8px 0;
        }
        .progress-bar-fill {
            height: 100%;
            border-radius: 6px;
            transition: width 0.8s ease;
        }

        /* ── Aviso de precisión ── */
        .precision-alert {
            padding: 12px 16px;
            border-radius: 10px;
            margin: 12px 0;
            font-size: 0.9em;
            border-left: 4px solid;
        }

        /* ── Títulos de sección ── */
        .section-title {
            color: #E0E0E0;
            font-size: 1.3em;
            font-weight: 600;
            margin: 24px 0 16px 0;
            padding-bottom: 8px;
            border-bottom: 1px solid #2D3142;
        }

        /* ── Nombre de documento en card ── */
        .doc-name {
            color: #E0E0E0;
            font-weight: 500;
            font-size: 0.95em;
        }

        /* ── Métricas de cobertura ── */
        .coverage-metric {
            text-align: center;
            padding: 12px;
        }
        .coverage-metric .value {
            font-size: 2.2em;
            font-weight: 700;
        }
        .coverage-metric .label {
            font-size: 0.85em;
            color: #8A8FA8;
            margin-top: 4px;
        }

        /* ── Sidebar: fondo oscuro y texto claro ── */
        [data-testid="stSidebar"] {
            background-color: #161b26;
        }
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span:not(.status-badge) {
            color: #E0E0E0 !important;
        }
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stNumberInput label {
            color: #C0C4D0 !important;
        }
        [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
            background-color: #1A1D27;
            border-color: #2D3142;
            color: #E0E0E0;
        }
        [data-testid="stSidebar"] .stNumberInput input {
            background-color: #1A1D27;
            color: #E0E0E0;
            border-color: #2D3142;
        }
        [data-testid="stSidebar"] hr {
            border-color: #2D3142;
        }

        /* ── Ajustes para Streamlit nativo en dark ── */
        .stTextInput, .stSelectbox, .stNumberInput {
            background-color: #1A1D27;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_document_checklist(cobertura: dict):
    """Renderiza el checklist visual de documentos requeridos usando componentes nativos de Streamlit."""
    st.markdown('<div class="section-title">📋 Checklist de Documentación Requerida</div>', unsafe_allow_html=True)

    for doc in DOCUMENTOS_REQUERIDOS:
        doc_id = doc["id"]
        detectado = cobertura.get(doc_id, False)

        if detectado:
            badge_class = "detected"
            badge_text = "✅ Detectado"
        elif cobertura:
            badge_class = "missing"
            badge_text = "❌ No detectado"
        else:
            badge_class = "pending"
            badge_text = "⏳ Pendiente"

        with st.container():
            # Fila principal: icono + nombre + tooltip | badge
            col_left, col_right = st.columns([5, 1])

            with col_left:
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center; gap: 12px; padding: 4px 0;">
                        <span style="font-size: 1.5em;">{doc['icono']}</span>
                        <span class="doc-name">{doc['nombre']}</span>
                        <div class="tooltip-container">
                            <span class="tooltip-icon">?</span>
                            <span class="tooltip-text">{doc['descripcion']}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col_right:
                st.markdown(
                    f'<div style="text-align: right; padding: 4px 0;">'
                    f'<span class="status-badge {badge_class}">{badge_text}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Línea separadora entre tarjetas
            st.markdown(
                "<hr style='margin: 2px 0; border: 0; border-top: 1px solid #2D3142; opacity: 0.5;'>",
                unsafe_allow_html=True,
            )


def render_coverage_dashboard(archivos: List):
    """Renderiza el dashboard de pre-análisis de cobertura documental."""
    if not archivos:
        return

    cobertura = analizar_cobertura(archivos)
    st.session_state.cobertura_documental = cobertura
    porcentaje = calcular_porcentaje(cobertura)
    nivel = obtener_nivel_cobertura(porcentaje)

    st.markdown('<div class="section-title">📊 Mapeo Documental — Pre-Análisis de Cobertura</div>', unsafe_allow_html=True)

    # ── Tarjeta principal de cobertura ──
    st.markdown(
        f"""
        <div class="coverage-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px;">
                <div style="flex: 1; min-width: 200px;">
                    <h3>📦 Cobertura del Dossier</h3>
                    <div class="coverage-metric">
                        <div class="value" style="color: {nivel['color']};">{porcentaje}%</div>
                        <div class="label">{nivel['label']}</div>
                    </div>
                    <div class="progress-bar-container">
                        <div class="progress-bar-fill" style="width: {porcentaje}%; background: {nivel['color']};"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.8em; color: #8A8FA8; margin-top: 4px;">
                        <span>0%</span>
                        <span>{sum(1 for v in cobertura.values() if v)}/{len(cobertura)} documentos</span>
                        <span>100%</span>
                    </div>
                </div>
                <div style="flex: 1; min-width: 200px;">
                    <h3>🎯 Precisión Esperada del Dictamen</h3>
                    <div style="margin-top: 12px;">
                        <div class="precision-alert" style="background: rgba({','.join(str(int(nivel['color'][i:i+2], 16)) for i in (1, 3, 5))}, 0.1); border-color: {nivel['color']};">
                            <div style="font-size: 1.3em; font-weight: 700; color: {nivel['color']};">{nivel['precision']}</div>
                            <div style="color: #A0A4B0; margin-top: 6px;">
                                {'✅ El dossier cuenta con documentación suficiente para una evaluación robusta.' if porcentaje >= 80 else
                                 '⚠️ La evaluación podrá realizarse, pero algunas dimensiones quedarán con supuestos.' if porcentaje >= 60 else
                                 '❌ La documentación es insuficiente. El dictamen tendrá baja confiabilidad.'}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Documentos identificados vs no identificados ──
    st.markdown("### 🔍 Detalle por Documento")
    render_document_checklist(cobertura)

    # ── Archivos no mapeados ──
    archivos_no_mapeados = []
    for archivo in archivos:
        doc_id = identificar_documento(archivo.name)
        if not doc_id:
            archivos_no_mapeados.append(archivo.name)

    if archivos_no_mapeados:
        with st.expander(f"📎 Archivos no clasificados ({len(archivos_no_mapeados)})"):
            for nombre in archivos_no_mapeados:
                st.caption(f"• {nombre}")
            st.info(
                "Estos archivos no pudieron ser clasificados automáticamente. "
                "Se procesarán igualmente, pero la cobertura mostrada puede no reflejar su contenido real."
            )


# ──────────────────────────────────────────────────────────────────────
#  CONFIGURACIÓN DE PÁGINA
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Evaluación Crediticia Chile",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()
render_dark_mode_css()


# ──────────────────────────────────────────────────────────────────────
#  SIDEBAR: CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────
#  CABECERA PRINCIPAL
# ──────────────────────────────────────────────────────────────────────
st.title("📊 Sistema de Evaluación Crediticia")
st.markdown(
    """
    Sistema multiagente para evaluación de **factoring** y **leasing** en Chile.
    Procesa carpetas tributarias y financieras para generar dictámenes de riesgo
    basados en políticas CMF y parámetros internos configurables.
    """
)

st.markdown("---")


# ──────────────────────────────────────────────────────────────────────
#  CHECKLIST DE DOCUMENTACIÓN (visible siempre)
# ──────────────────────────────────────────────────────────────────────
render_document_checklist(st.session_state.cobertura_documental)

st.markdown("---")


# ──────────────────────────────────────────────────────────────────────
#  ZONA DE CARGA INTELIGENTE (Multi-file)
# ──────────────────────────────────────────────────────────────────────
st.header("📄 Zona de Carga Inteligente")

col_upload, col_status = st.columns([3, 1])

with col_upload:
    archivos_subidos = st.file_uploader(
        "Arrastra o selecciona los documentos del cliente",
        type=["pdf", "zip"],
        accept_multiple_files=True,
        help=(
            "Puedes subir múltiples archivos PDF individuales o un ZIP consolidado. "
            "El sistema identificará automáticamente cada documento por su nombre."
        ),
    )

with col_status:
    st.markdown("### Estado")
    if archivos_subidos:
        st.success(f"✅ {len(archivos_subidos)} archivo(s) cargado(s)")
    else:
        st.info("⏳ Sin archivos")

# ── Dashboard de cobertura (se muestra tras la carga) ──
if archivos_subidos:
    st.session_state.archivos_subidos = archivos_subidos
    render_coverage_dashboard(archivos_subidos)
else:
    st.session_state.archivos_subidos = []
    st.session_state.cobertura_documental = {}

st.markdown("---")


# ──────────────────────────────────────────────────────────────────────
#  BOTÓN DE EVALUACIÓN
# ──────────────────────────────────────────────────────────────────────
if archivos_subidos:
    st.session_state.pdf_procesado = True

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
            # ── Consolidar múltiples PDFs en uno solo ──────────────
            # Si hay múltiples archivos, los concatenamos en un PDF temporal
            # (o usamos el primero si es ZIP consolidado)
            pdf_paths = []

            with status_placeholder:
                with etapas:
                    paso_ingestion.write("🔄 1/4  Ingestion — preparando documentos…")

            for archivo in archivos_subidos:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{archivo.name}") as tmp:
                    tmp.write(archivo.getvalue())
                    pdf_paths.append(tmp.name)

            # Usar el primer PDF como principal (o el ZIP si corresponde)
            pdf_path = pdf_paths[0] if pdf_paths else None

            if not pdf_path:
                raise ValueError("No se pudo preparar ningún archivo para la evaluación.")

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

            # Limpiar archivos temporales
            for p in pdf_paths:
                try:
                    os.unlink(p)
                except OSError:
                    pass

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
        st.session_state.archivos_subidos = []
        st.session_state.cobertura_documental = {}
        st.rerun()


# ──────────────────────────────────────────────────────────────────────
#  VISUALIZACIÓN DE RESULTADOS
# ──────────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────
#  MANEJO DE ERRORES
# ──────────────────────────────────────────────────────────────────────
if st.session_state.error:
    st.error(f"❌ Error en la evaluación: {st.session_state.error}")
    if st.button("Reintentar"):
        st.session_state.error = None
        st.session_state.resultado = None
        st.session_state.pdf_procesado = False
        st.session_state.etapa_actual = "carga"
        st.session_state.archivos_subidos = []
        st.session_state.cobertura_documental = {}
        st.rerun()


# ──────────────────────────────────────────────────────────────────────
#  FOOTER
# ──────────────────────────────────────────────────────────────────────
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