"""Aplicación principal Streamlit para evaluación crediticia multiagente.

Conecta la interfaz de usuario con el grafo LangGraph
(``ejecutar_evaluacion`` de ``src.state.graph_state``).
"""

import json
import os
import tempfile
from typing import Optional, List

import streamlit as st
import streamlit.components.v1 as components

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


def render_header_banner():
    """Renderiza el banner corporativo superior con branding PRO."""
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #0F1420 0%, #1A1F35 50%, #0F1420 100%);
            border: 1px solid #2A3050;
            border-radius: 16px;
            padding: 22px 32px;
            margin-bottom: 20px;
            display: flex;
            flex-direction: row;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 4px 24px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255,255,255,0.04);
        ">
            <div style="display: flex; align-items: center; gap: 18px;">
                <div style="
                    background: linear-gradient(135deg, #00C9B7 0%, #0099FF 100%);
                    width: 44px; height: 44px;
                    border-radius: 12px;
                    display: flex; align-items: center; justify-content: center;
                    font-size: 22px;
                    box-shadow: 0 4px 12px rgba(0, 153, 255, 0.3);
                ">🏦</div>
                <div>
                    <div style="
                        font-size: 0.7em;
                        font-weight: 700;
                        letter-spacing: 3px;
                        color: #00C9B7;
                        text-transform: uppercase;
                        margin-bottom: 2px;
                    ">PRO — Enterprise Credit Evaluation</div>
                    <div style="
                        font-size: 0.85em;
                        color: #8890B0;
                        font-weight: 300;
                    ">Chile · CMF Compliance Framework · v2.0</div>
                </div>
            </div>
            <div style="
                display: flex;
                align-items: center;
                gap: 12px;
                background: rgba(0, 153, 255, 0.08);
                padding: 8px 18px;
                border-radius: 30px;
                border: 1px solid rgba(0, 153, 255, 0.15);
            ">
                <span style="color: #00C9B7; font-size: 1.1em;">🔒</span>
                <span style="color: #C0C8E0; font-size: 0.78em; font-weight: 400;">
                    Secure Multi-Agent Risk Engine
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_producto_info_card(producto: str):
    """Renderiza tarjeta visual dinámica con SVG inline y resumen de parámetros CMF."""
    if producto == "factoring":
        svg_ilustracion = """\
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 160" width="240" height="160">
            <defs>
                <linearGradient id="factGrad1" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#00C9B7;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#0099FF;stop-opacity:1" />
                </linearGradient>
                <linearGradient id="factGrad2" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" style="stop-color:#1A2A40;stop-opacity:0.9" />
                    <stop offset="100%" style="stop-color:#0F1420;stop-opacity:0.9" />
                </linearGradient>
            </defs>
            <rect x="10" y="20" width="60" height="40" rx="4" fill="url(#factGrad2)" stroke="#2A4060" stroke-width="1.2"/>
            <rect x="20" y="28" width="18" height="12" rx="2" fill="none" stroke="#00C9B7" stroke-width="1.2"/>
            <rect x="42" y="28" width="18" height="12" rx="2" fill="none" stroke="#00C9B7" stroke-width="1.2"/>
            <line x1="12" y1="50" x2="68" y2="50" stroke="#2A4060" stroke-width="1"/>
            <line x1="25" y1="50" x2="25" y2="54" stroke="#0099FF" stroke-width="1.5"/>
            <line x1="40" y1="50" x2="40" y2="54" stroke="#0099FF" stroke-width="1.5"/>
            <line x1="55" y1="50" x2="55" y2="54" stroke="#0099FF" stroke-width="1.5"/>
            <!-- Flecha flujo de caja -->
            <path d="M85 40 L110 40 L110 30 L130 45 L110 60 L110 50 L85 50 Z" fill="#00C9B7" opacity="0.7"/>
            <circle cx="145" cy="45" r="6" fill="url(#factGrad1)" opacity="0.6"/>
            <!-- Documentos / billetes -->
            <rect x="155" y="30" width="50" height="38" rx="3" fill="#1A2A40" stroke="#2A4060" stroke-width="1"/>
            <rect x="160" y="36" width="30" height="4" rx="2" fill="#0099FF" opacity="0.5"/>
            <rect x="160" y="44" width="24" height="4" rx="2" fill="#0099FF" opacity="0.3"/>
            <rect x="160" y="52" width="26" height="4" rx="2" fill="#0099FF" opacity="0.4"/>
            <!-- Gráfico ascendente -->
            <polyline points="10,110 30,100 50,105 70,90 90,92 110,75 130,78 150,65 170,68 190,55 210,50 230,45"
                      fill="none" stroke="#00C9B7" stroke-width="2" opacity="0.6"/>
            <polyline points="10,130 30,125 50,128 70,118 90,120 110,108 130,110 150,100 170,102 190,92 210,88 230,82"
                      fill="none" stroke="#0099FF" stroke-width="1.5" opacity="0.4"/>
            <circle cx="230" cy="45" r="3" fill="#00C9B7"/>
            <circle cx="230" cy="82" r="2.5" fill="#0099FF"/>
            <text x="120" y="148" text-anchor="middle" fill="#00C9B7" font-size="8" font-family="monospace">LIQUIDEZ · FLUJO DE CAJA</text>
        </svg>"""
        parametros_cmf = """\
        <div style="margin-top: 10px; font-size: 0.85em; line-height: 1.6; color: #C0C8E0;">
            <div style="color: #00C9B7; font-weight: 600; margin-bottom: 6px;">📌 Parámetros CMF priorizados para <strong>Factoring</strong>:</div>
            <ul style="margin: 0; padding-left: 18px;">
                <li>🔹 <strong>Rotación de Cartera</strong> — días promedio de cobro</li>
                <li>🔹 <strong>Concentración de Deudores</strong> — riesgo de dependencia</li>
                <li>🔹 <strong>Liquidez Corriente</strong> — capacidad de pago inmediata</li>
                <li>🔹 <strong>Endeudamiento Total</strong> — apalancamiento financiero</li>
                <li>🔹 <strong>Calidad de Documentos Comerciales</strong> — antigüedad y tipo</li>
            </ul>
            <div style="margin-top: 8px; padding: 6px 10px; background: rgba(0,201,183,0.08); border-radius: 6px; border-left: 3px solid #00C9B7;">
                ⚡ Evaluación centrada en liquidez, morosidad histórica y calidad de la cartera cedible.
            </div>
        </div>"""
    else:  # leasing
        svg_ilustracion = """\
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 160" width="240" height="160">
            <defs>
                <linearGradient id="leaseGrad1" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#FF8C42;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#FF5E62;stop-opacity:1" />
                </linearGradient>
                <linearGradient id="leaseGrad2" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" style="stop-color:#1A2A40;stop-opacity:0.9" />
                    <stop offset="100%" style="stop-color:#0F1420;stop-opacity:0.9" />
                </linearGradient>
            </defs>
            <!-- Camión / vehículo industrial -->
            <rect x="20" y="60" width="80" height="45" rx="6" fill="#1A2A40" stroke="#3A5070" stroke-width="1.2"/>
            <rect x="40" y="70" width="40" height="22" rx="3" fill="none" stroke="#3A5070" stroke-width="1"/>
            <line x1="40" y1="80" x2="80" y2="80" stroke="#3A5070" stroke-width="0.8"/>
            <line x1="60" y1="70" x2="60" y2="92" stroke="#3A5070" stroke-width="0.8"/>
            <!-- Ruedas -->
            <circle cx="40" cy="110" r="10" fill="#0F1420" stroke="#3A5070" stroke-width="1.5"/>
            <circle cx="40" cy="110" r="4" fill="#FF8C42" opacity="0.6"/>
            <circle cx="80" cy="110" r="10" fill="#0F1420" stroke="#3A5070" stroke-width="1.5"/>
            <circle cx="80" cy="110" r="4" fill="#FF8C42" opacity="0.6"/>
            <!-- Maquinaria / engranaje -->
            <circle cx="145" cy="60" r="20" fill="none" stroke="#FF8C42" stroke-width="3" opacity="0.5"/>
            <circle cx="145" cy="60" r="8" fill="#FF5E62" opacity="0.4"/>
            <line x1="145" y1="40" x2="145" y2="80" stroke="#FF8C42" stroke-width="1.5" opacity="0.3"/>
            <line x1="125" y1="60" x2="165" y2="60" stroke="#FF8C42" stroke-width="1.5" opacity="0.3"/>
            <!-- Edificio / activo fijo -->
            <rect x="120" y="90" width="50" height="50" rx="2" fill="#1A2A40" stroke="#3A5070" stroke-width="1"/>
            <rect x="130" y="98" width="10" height="14" rx="1" fill="#FF8C42" opacity="0.3"/>
            <rect x="150" y="98" width="10" height="14" rx="1" fill="#FF8C42" opacity="0.3"/>
            <rect x="130" y="118" width="10" height="14" rx="1" fill="#FF8C42" opacity="0.3"/>
            <rect x="150" y="118" width="10" height="14" rx="1" fill="#FF8C42" opacity="0.3"/>
            <text x="120" y="148" text-anchor="middle" fill="#FF8C42" font-size="8" font-family="monospace">ACTIVOS FIJOS · MAQUINARIA · VEHÍCULOS</text>
        </svg>"""
        parametros_cmf = """\
        <div style="margin-top: 10px; font-size: 0.85em; line-height: 1.6; color: #C0C8E0;">
            <div style="color: #FF8C42; font-weight: 600; margin-bottom: 6px;">📌 Parámetros CMF priorizados para <strong>Leasing</strong>:</div>
            <ul style="margin: 0; padding-left: 18px;">
                <li>🔸 <strong>Cobertura EBITDA</strong> — capacidad de pago del arriendo</li>
                <li>🔸 <strong>Cobertura Servicio Deuda (FCF)</strong> — flujo libre disponible</li>
                <li>🔸 <strong>Endeudamiento Máximo</strong> — nivel de apalancamiento</li>
                <li>🔸 <strong>Margen EBITDA</strong> — rentabilidad operacional</li>
                <li>🔸 <strong>Vida Útil del Activo</strong> — relación plazo/garantía</li>
            </ul>
            <div style="margin-top: 8px; padding: 6px 10px; background: rgba(255,140,66,0.08); border-radius: 6px; border-left: 3px solid #FF8C42;">
                ⚡ Evaluación centrada en generación de flujo, solvencia patrimonial y capacidad de pago recurrente.
            </div>
        </div>"""

    # Tarjeta combinada - HTML completamente inline
    html_content = f"""\
        <div style="
            background: linear-gradient(135deg, #111827 0%, #1A1F35 100%);
            border: 1px solid #2A3050;
            border-radius: 16px;
            padding: 20px;
            margin: 16px 0;
            transition: all 0.2s;
        ">
            <div style="display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-start;">
                <div style="flex-shrink: 0;">
                    {svg_ilustracion}
                </div>
                <div style="flex: 1; min-width: 180px;">
                    {parametros_cmf}
                </div>
            </div>
        </div>"""
    components.html(html_content, height=350)


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

        /* ── Header superior transparente / oscuro ── */
        header[data-testid="stHeader"] {
            background-color: #0e1117 !important;
            border-bottom: 1px solid #1E2230;
        }

        /* ── Zona de carga (file uploader) oscura ── */
        [data-testid="stFileUploader"] {
            background-color: #161b26;
            border: 1px solid #2D3142;
            border-radius: 12px;
            padding: 8px;
            transition: border-color 0.2s;
        }
        [data-testid="stFileUploader"]:hover {
            border-color: #4A4F6A;
        }
        [data-testid="stFileUploader"] section {
            background-color: #161b26;
        }
        [data-testid="stFileUploader"] button {
            background-color: #2D3142;
            color: #E0E0E0;
            border: 1px solid #4A4F6A;
        }
        [data-testid="stFileUploader"] button:hover {
            background-color: #3D4260;
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

        /* ── HEADER: barra superior transparente ── */
        header[data-testid="stHeader"] {
            background-color: #0e1117 !important;
            border-bottom: 1px solid #1E2235;
        }
        header[data-testid="stHeader"] * {
            color: #8A8FA8 !important;
        }

        /* ── FILE UPLOADER: fondo oscuro alineado con sidebar ── */
        section[data-testid="stFileUploader"] {
            background-color: #161b26;
            border: 1px dashed #2D3142;
            border-radius: 12px;
            padding: 8px;
            transition: border-color 0.2s;
        }
        section[data-testid="stFileUploader"]:hover {
            border-color: #4A4F6A;
        }
        section[data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] p {
            color: #C0C4D0 !important;
        }
        section[data-testid="stFileUploader"] button {
            background-color: #1A1D27 !important;
            border: 1px solid #2D3142 !important;
            color: #E0E0E0 !important;
        }
        section[data-testid="stFileUploader"] button:hover {
            border-color: #4A4F6A !important;
        }
        .stFileUploaderFile {
            background-color: #1A1D27 !important;
            color: #E0E0E0 !important;
            border: 1px solid #2D3142 !important;
            border-radius: 8px !important;
        }
        .stFileUploaderFile * {
            color: #E0E0E0 !important;
        }

        /* ── Banner de cabecera corporativo ── */
        .corp-banner {
            background: linear-gradient(135deg, #0F1320 0%, #1A1F3A 50%, #0F1923 100%);
            border: 1px solid #2A3050;
            border-radius: 16px;
            padding: 20px 28px;
            margin-bottom: 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 12px;
            position: relative;
            overflow: hidden;
        }
        .corp-banner::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -20%;
            width: 200px;
            height: 200px;
            background: radial-gradient(circle, rgba(59, 130, 246, 0.08) 0%, transparent 70%);
            pointer-events: none;
        }
        .corp-banner::after {
            content: '';
            position: absolute;
            bottom: -30%;
            right: -10%;
            width: 150px;
            height: 150px;
            background: radial-gradient(circle, rgba(0, 204, 102, 0.06) 0%, transparent 70%);
            pointer-events: none;
        }
        .corp-banner .banner-tag {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(0, 204, 102, 0.1));
            border: 1px solid rgba(59, 130, 246, 0.25);
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 0.75em;
            font-weight: 700;
            letter-spacing: 1.5px;
            color: #60A5FA;
            text-transform: uppercase;
        }
        .corp-banner .banner-tag .dot {
            width: 6px;
            height: 6px;
            background: #00CC66;
            border-radius: 50%;
            animation: pulse-dot 2s infinite;
        }
        @keyframes pulse-dot {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.5; transform: scale(0.8); }
        }
        .corp-banner .banner-title {
            font-size: 1.05em;
            font-weight: 600;
            color: #F0F4FF;
            position: relative;
            z-index: 1;
        }
        .corp-banner .banner-sub {
            font-size: 0.8em;
            color: #8A8FA8;
            position: relative;
            z-index: 1;
        }

        /* ── Tarjetas de producto dinámicas ── */
        .product-card {
            background: linear-gradient(135deg, #1A1D27 0%, #1F2340 100%);
            border: 1px solid #2D3142;
            border-radius: 16px;
            padding: 20px;
            margin: 16px 0;
            transition: all 0.3s ease;
        }
        .product-card:hover {
            border-color: #4A4F6A;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
        }
        .product-card .product-visual {
            text-align: center;
            margin-bottom: 16px;
        }
        .product-card .product-visual svg {
            max-width: 100%;
            height: auto;
        }
        .product-card .product-params {
            background: rgba(14, 17, 23, 0.6);
            border: 1px solid #2D3142;
            border-radius: 10px;
            padding: 14px;
            margin-top: 12px;
        }
        .product-card .product-params h4 {
            color: #E0E0E0;
            font-size: 0.85em;
            margin: 0 0 8px 0;
        }
        .product-card .product-params ul {
            list-style: none;
            padding: 0;
            margin: 0;
        }
        .product-card .product-params ul li {
            padding: 4px 0;
            font-size: 0.82em;
            color: #A0A4B0;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .product-card .product-params ul li::before {
            content: '▸';
            color: #60A5FA;
            font-weight: bold;
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

    # ── Tarjeta visual dinámica del producto seleccionado ──
    render_producto_info_card(producto_seleccionado)

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

# Banner corporativo con avatar
col_texto, col_avatar = st.columns([3, 1])

with col_texto:
    st.markdown(
        f"""
        <div class="corp-banner">
            <div>
                <div class="banner-tag">
                    <span class="dot"></span>
                    PRO — ENTERPRISE CREDIT EVALUATION
                </div>
                <div class="banner-title" style="margin-top: 8px;">
                    📊 Sistema de Evaluación Crediticia
                </div>
                <div class="banner-sub">
                    Multiagente · Chile · CMF Compliance
                </div>
            </div>
            <div style="text-align: right; position: relative; z-index: 1;">
                <div style="font-size: 2em; font-weight: 700; color: #60A5FA; line-height: 1;">
                    {producto_seleccionado.upper()}
                </div>
                <div style="font-size: 0.75em; color: #8A8FA8; letter-spacing: 2px; text-transform: uppercase;">
                    Producto Activo
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_avatar:
    try:
        st.image("assets/agente.png", width='stretch')
    except FileNotFoundError:
        pass
    st.markdown(
        """
        <style>
        [data-testid="column"]:nth-child(2) img {
            border-radius: 16px;
            box-shadow: 0 0 20px rgba(59, 130, 246, 0.3), 0 0 60px rgba(59, 130, 246, 0.1);
            transition: box-shadow 0.3s ease;
        }
        [data-testid="column"]:nth-child(2) img:hover {
            box-shadow: 0 0 30px rgba(59, 130, 246, 0.5), 0 0 80px rgba(59, 130, 246, 0.2);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    Sistema multiagente para evaluación de **factoring** y **leasing** en Chile.\n
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
