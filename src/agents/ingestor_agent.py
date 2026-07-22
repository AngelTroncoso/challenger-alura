"""Agente Ingestor: especializado en extracción de datos desde PDFs del SII."""

from typing import Any, Dict, Optional
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseLanguageModel
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.tools import Tool
from src.tools.pdf_parser_tool import (
    extraer_texto,
    extraer_formulario_29,
    extraer_formulario_22,
    extraer_balance,
    extraer_dicom,
    procesar_dossier,
)
from src.models.schemas import InfoTributaria, EstadosFinancieros, InfoCrediticia
from src.utils.validators import validar_rut


class IngestorAgent:
    """Agente encargado de ingerir y estructurar datos desde PDFs tributarios.

    Este agente utiliza PyPDF para extraer texto de documentos del SII
    (Formularios 29, 22, balances, informes DICOM) y los estructura
    en modelos Pydantic para su procesamiento posterior.
    """

    def __init__(self, llm: BaseLanguageModel):
        """Inicializa el agente con un modelo de lenguaje.

        Args:
            llm: Modelo de lenguaje para interpretación de textos.
        """
        self.llm = llm
        self.tools = self._crear_herramientas()
        self.agent = self._crear_agente()

    def _crear_herramientas(self) -> list:
        """Crea las herramientas disponibles para este agente.

        Returns:
            Lista de herramientas LangChain.
        """
        return [
            Tool(
                name="extraer_texto_pdf",
                func=extraer_texto,
                description=(
                    "Extrae todo el texto plano de un archivo PDF. "
                    "Útil para obtener el contenido crudo de documentos tributarios. "
                    "Input: ruta completa al archivo PDF."
                ),
            ),
            Tool(
                name="extraer_formulario29",
                func=lambda texto: str(extraer_formulario_29(texto)),
                description=(
                    "Extrae datos del Formulario 29 (declaración mensual de IVA) "
                    "desde el texto extraído del PDF. "
                    "Input: texto plano del PDF."
                ),
            ),
            Tool(
                name="extraer_formulario22",
                func=lambda texto: str(extraer_formulario_22(texto)),
                description=(
                    "Extrae datos del Formulario 22 (declaración anual de renta) "
                    "desde el texto extraído del PDF. "
                    "Input: texto plano del PDF."
                ),
            ),
            Tool(
                name="extraer_balance",
                func=lambda texto: str(extraer_balance(texto)),
                description=(
                    "Extrae datos del balance 8 columnas y estado de resultados "
                    "desde el texto extraído del PDF. "
                    "Input: texto plano del PDF."
                ),
            ),
            Tool(
                name="extraer_dicom",
                func=lambda texto: str(extraer_dicom(texto)),
                description=(
                    "Extrae información de protestos y morosidades "
                    "desde informe DICOM/Platinum. "
                    "Input: texto plano del PDF."
                ),
            ),
            Tool(
                name="procesar_dossier_completo",
                func=procesar_dossier,
                description=(
                    "Orquesta la extracción completa de un dossier PDF "
                    "identificando automáticamente los tipos de documento. "
                    "Input: ruta completa al archivo PDF."
                ),
            ),
        ]

    def _crear_agente(self):
        """Crea el agente LangChain con prompt especializado.

        Returns:
            AgentExecutor configurado.
        """
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Eres un Ingestor de Documentos Financieros especializado en el SII chileno.\n\n"
                "Tu función es:\n"
                "1. Extraer texto de PDFs usando la herramienta extraer_texto_pdf\n"
                "2. Identificar qué tipo de documento es (F29, F22, Balance, DICOM)\n"
                "3. Aplicar el extractor específico según el tipo de documento\n"
                "4. Estructurar los datos extraídos\n\n"
                "REGLAS:\n"
                "- Usa la herramienta 'procesar_dossier_completo' si el PDF contiene múltiples documentos\n"
                "- Valida RUTs cuando encuentres RUT de empresa\n"
                "- Maneja formatos numéricos chilenos (puntos como separador de miles, coma decimal)\n"
                "- Si encuentras errores de extracción, indícalos claramente\n\n"
                "Siempre reporta exactamente qué documentos encontraste y qué datos pudiste extraer."
            ),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_structured_chat_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt,
        )

        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10,
        )

    def procesar(self, pdf_path: str) -> Dict[str, Any]:
        """Ejecuta el agente para procesar un PDF de dossier.

        Args:
            pdf_path: Ruta al archivo PDF del dossier.

        Returns:
            Diccionario con los datos estructurados extraídos.
        """
        resultado = self.agent.invoke({"input": f"Procesa el dossier PDF en {pdf_path}"})
        return resultado