# Implementation Plan

[Overview]
Desarrollar un sistema multiagente de evaluación crediticia para factoring y leasing en Chile, utilizando LangChain/LangGraph, con interfaz Streamlit, que procesa PDFs de carpetas tributarias y financieras para generar dictámenes de riesgo basados en políticas CMF y parámetros internos configurables.

El sistema implementará una arquitectura de 4 agentes especializados (Ingestor, Analista Financiero, Evaluador de Riesgo, Dictaminador) que procesan información del SII (formularios 29 y 22), balances, estados de resultados, informes DICOM/Platinum y metadatos comerciales. Incluirá motor de cálculo de ratios financieros específicos para factoring (riesgo del deudor, liquidez corto plazo) y leasing (capacidad de pago, cobertura EBITDA, garantías), con persistencia en memoria inicialmente y arquitectura preparada para integración futura con PostgreSQL/Supabase.

[Types]
Definir estructuras de datos para el dominio crediticio chileno, incluyendo tipos para documentos tributarios, ratios financieros, políticas parametrizables y resultados de evaluación.

```python
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

# Información tributaria del SII
class InfoTributaria(BaseModel):
    rut_empresa: str
    razon_social: str
    fecha_inicio_actividades: Optional[date]
    formulario29_ultimos_12m: List[Dict[str, float]]  # mes, iva_ventas, iva_compras, impuesto_pagado
    formulario22_anual: Dict[str, float]  # renta_ liquida, impuesto_pagado, capital_propio

# Estados financieros
class EstadosFinancieros(BaseModel):
    periodo: str
    activos_corrientes: float
    activos_no_corrientes: float
    pasivos_corrientes: float
    pasivos_no_corrientes: float
    patrimonio: float
    ventas_netas: float
    costo_ventas: float
    resultado_operacional: float
    resultado_neto: float
    depreciacion: float
    gastos_financieros: float

# Información crediticia/comercial
class InfoCrediticia(BaseModel):
    tiene_protestos: bool
    monto_protestos: float
    tiene_morosidades: bool
    dias_morosidad: int
    morosidades_previsionales: bool
    antiguedad_meses: int

# Ratios financieros calculados
class RatiosFinancieros(BaseModel):
    liquidez_corriente: float
    liquidez_inmediata: float
    razon_deuda_patrimonio: float
    endeudamiento_total: float
    cobertura_servicio_deuda_fcf: float  # Para leasing
    margen_ebitda: float
    roi: float
    rotacion_cartera: float  # Para factoring

# Parámetros de política configurables
class PoliticasEvaluacion(BaseModel):
    producto: TipoProducto
    antiguedad_minima_meses: int = 12
    liquidez_minima: float = 1.2
    endeudamiento_maximo: float = 2.5
    cobertura_ebitda_minima: float = 1.5
    morosidades_permitidas: bool = False
    dias_morosidad_maximo: int = 0

# Resultado de evaluación
class ResultadoEvaluacion(BaseModel):
    rut_empresa: str
    producto: TipoProducto
    estado: EstadoEvaluacion
    puntaje_riesgo: float  # 0-100
    ratios_calculados: RatiosFinancieros
    politicas_aplicadas: PoliticasEvaluacion
    dictamen: str
    condiciones_aplicables: List[str]
    factores_positivos: List[str]
    factores_riesgo: List[str]
    recomendaciones: List[str]
```

[Files]
Crear estructura de proyecto modular con separación de agentes, herramientas, utilidades y configuración, desplegable en Streamlit Cloud desde GitHub.

Nuevos archivos:
- `pyproject.toml` - Configuración de dependencias (langchain, langgraph, pypdf, pandas, streamlit, groq)
- `.streamlit/config.toml` - Configuración de tema y página de Streamlit
- `src/agents/__init__.py` - Inicialización del paquete de agentes
- `src/agents/ingestor_agent.py` - Agente especializado en extracción de PDFs con PyPDF y estructuración de datos
- `src/agents/analista_financiero.py` - Agente de cálculo de ratios con Pandas
- `src/agents/evaluador_riesgo.py` - Agente de aplicación de políticas y scoring
- `src/agents/dictaminador.py` - Agente orquestador y generador de dictámenes
- `src/tools/pdf_parser_tool.py` - Tool personalizada para extracción de texto/tablas desde PDFs del SII
- `src/tools/ratio_calculator_tool.py` - Tool para cálculo financiero (liquidez, endeudamiento, coberturas)
- `src/tools/policy_engine_tool.py` - Tool para validación de políticas CMF e internas
- `src/utils/validators.py` - Validadores de RUT, formatos tributarios chilenos
- `src/utils/config.py` - Carga de configuraciones y variables de entorno
- `src/models/schemas.py` - Modelos Pydantic para tipos de dominio
- `src/state/graph_state.py` - Definición del estado compartido LangGraph
- `app.py` - Aplicación principal Streamlit con UI de carga, selección de producto y visualización de resultados
- `requirements.txt` - Dependencias Python para deployment
- `.env.example` - Plantilla de variables de entorno (GROQ_API_KEY)

Archivos a modificar:
- Ninguno (proyecto nuevo)

[Functions]
Implementar funciones específicas para procesamiento de documentos chilenos, cálculo de ratios normativos y generación de dictámenes.

Nuevas funciones:
- `src/tools/pdf_parser_tool.py::extraer_formulario_29(text: str) -> InfoTributaria`
  Extrae declaraciones mensuales de IVA del PDF del SII
  
- `src/tools/pdf_parser_tool.py::extraer_formulario_22(text: str) -> InfoTributaria`
  Extrae declaración anual de renta del PDF del SII
  
- `src/tools/pdf_parser_tool.py::extraer_balance(text: str) -> EstadosFinancieros`
  Extrae balance 8 columnas y estado de resultados
  
- `src/tools/pdf_parser_tool.py::extraer_dicom(text: str) -> InfoCrediticia`
  Extrae información de protestos y morosidades
  
- `src/tools/ratio_calculator_tool.py::calcular_ratios_factoring(ee: EstadosFinancieros, info_cred: InfoCrediticia) -> RatiosFinancieros`
  Calcula ratios específicos para factoring (rotación de cartera, liquidez)
  
- `src/tools/ratio_calculator_tool.py::calcular_ratios_leasing(ee: EstadosFinancieros) -> RatiosFinancieros`
  Calcula ratios específicos para leasing (cobertura EBITDA, endeudamiento)
  
- `src/tools/policy_engine_tool.py::evaluar_cumplimiento(ratios: RatiosFinancieros, politicas: PoliticasEvaluacion) -> Dict`
  Valida políticas CMF e internas, retorna lista de incumplimientos
  
- `src/agents/ingestor_agent.py::procesar_dossier(pdf_path: str) -> Dict`
  Orquesta extracción completa del dossier PDF
  
- `src/agents/dictaminador.py::generar_dictamen(resultados: Dict) -> str`
  Genera dictamen natural basado en evaluación

[Classes]
Clases principales del sistema multiagente con LangGraph.

Nuevas clases:
- `src/agents/ingestor_agent.py::IngestorAgent(Agent)` 
  Agente con herramientas de PyPDF para extracción de documentos tributarios SII, balances e informes comerciales. Prompt enfocado en identificación y segmentación de tablas financieras.
  
- `src/agents/analista_financiero.py::AnalistaFinancieroAgent(Agent)`
  Agente con herramientas de Pandas para cálculos de ratios, normalización de datos financieros y generación de métricas. Prompt con fórmulas chilenas estándar.
  
- `src/agents/evaluador_riesgo.py::EvaluadorRiesgoAgent(Agent)`
  Agente con herramienta policy_engine para validación de políticas CMF e internas. Prompt con criterios de clasificación de riesgo CMF.
  
- `src/agents/dictaminador.py::DictaminadorAgent(Agent)`
  Agente orquestador que consume resultados de los demás agentes y genera dictamen comercial final. Prompt con estructura formal de informe crediticio.
  
- `src/state/graph_state.py::GraphState(TypedDict)`
  Estado compartido del grafo LangGraph con campos para documentos, ratios, evaluación y dictamen.

[Dependencies]
Gestionar dependencias con separación entre runtime, desarrollo y deployment, priorizando compatibilidad con Groq API y Streamlit Cloud.

Agregar:
- `langchain>=0.2.0` - Orquestación de agentes y prompts
- `langchain-experimental>=0.0.60` - Soporte para herramientas personalizadas
- `langgraph>=0.2.0` - Construcción del grafo multiagente
- `pypdf2>=3.0.0` o `pypdf>=4.0.0` - Extracción de texto y tablas desde PDFs
- `pandas>=2.0.0` - Cálculo de ratios y manipulación de datos financieros
- `streamlit>=1.30.0` - Interfaz web
- `groq>=0.9.0` - Cliente API para LLM (modelos Llama/Groq)
- `python-dotenv>=1.0.0` - Variables de entorno
- `pydantic>=2.0.0` - Validación de datos
- `plotly>=5.17.0` - Gráficos interactivos para Streamlit
- `tabulate>=0.9.0` - Formateo de tablas para reportes

[Testing]
Validar parsing de PDFs chilenos, exactitud de cálculos financieros, cumplimiento de políticas y generación de dictámenes.

Test files:
- `tests/test_pdf_parser.py` - Tests unitarios de extracción con PDFs de ejemplo (Formulario 29, 22, balances)
- `tests/test_ratio_calculator.py` - Tests de fórmulas financieras con valores conocidos
- `tests/test_policy_engine.py` - Tests de validación de políticas con escenarios de aprobación/rechazo
- `tests/test_agents_integration.py` - Test de integración del grafo completo con dossier de prueba
- `tests/conftest.py` - Fixtures con datos de ejemplo (PDFs ficticios, balances de prueba)

Estrategia:
- Tests unitarios por herramienta con mocks de PDFs cuando no haya datos reales
- Validación de fórmulas financieras contra cálculos manuales
- Casos de prueba para factoring (corto plazo, riesgo deudor) y leasing (mediano plazo, capacidad de pago)
- Verificación de formato de dictámenes y campos obligatorios

[Implementation Order]
Secuencia de implementación enfocada en capas base primero, luego agentes, luego orquestación y finalmente UI.

1. Inicializar proyecto con estructura de carpetas y configuraciones base (pyproject.toml, requirements.txt, .env.example, .streamlit/config.toml)
2. Implementar modelos/datos: schemas.py con todos los modelos Pydantic definidos, validators.py con RUT y formatos chilenos, config.py con carga de variables
3. Crear herramientas base: pdf_parser_tool.py con extracción de Formularios 29/22, balance y DICOM; ratio_calculator_tool.py con 8 ratios financieros; policy_engine_tool.py con motor de validación parametrizable
4. Implementar agentes individuales: IngestorAgent, AnalistaFinancieroAgent, EvaluadorRiesgoAgent, DictaminadorAgent con sus respectivos prompts y tools
5. Construir grafo LangGraph en graph_state.py conectando los 4 agentes en flujo secuencial
6. Desarrollar interfaz Streamlit en app.py con carga de PDF, selección de producto (factoring/leasing), visualización de ratios en tablas/gráficos y descarga de dictamen
7. Configurar deployment: conectar con Groq API, preparar secrets para Streamlit Cloud, crear README.md con instrucciones
8. Testing unitario e integración: tests por módulo, fixture de PDFs de ejemplo, validación completa del flujo