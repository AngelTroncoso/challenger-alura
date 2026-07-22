"""Agente Dictaminador: orquestador y generador de dictámenes crediticios."""

from typing import Any, Dict, List
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseLanguageModel
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.tools import Tool
from src.models.schemas import (
    ResultadoEvaluacion,
    RatiosFinancieros,
    PoliticasEvaluacion,
    InfoTributaria,
    InfoCrediticia,
    EstadosFinancieros,
    EstadoEvaluacion,
    TipoProducto,
)


class DictaminadorAgent:
    """Agente orquestador que genera el dictamen crediticio final.

    Este agente consume los resultados de los agentes anteriores
    (Ingestor, Analista, Evaluador) y genera un dictamen comercial
    estructurado con formato de informe crediticio profesional.
    """

    def __init__(self, llm: BaseLanguageModel):
        """Inicializa el agente con un modelo de lenguaje.

        Args:
            llm: Modelo de lenguaje para generación de dictámenes.
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
                name="estructurar_resultado",
                func=self._estructurar_resultado,
                description=(
                    "Estructura el resultado completo de la evaluación en un objeto ResultadoEvaluacion. "
                    "Requiere todos los datos de la evaluación en formato JSON."
                ),
            ),
        ]

    def _estructurar_resultado(
        self,
        rut: str,
        producto: str,
        estado: str,
        puntaje: float,
        ratios_json: str,
        politicas_json: str,
        dictamen: str,
        condiciones: str,
        factores_positivos: str,
        factores_riesgo: str,
        recomendaciones: str,
    ) -> str:
        """Estructura el resultado en un modelo Pydantic.

        Args:
            rut: RUT de la empresa.
            producto: Tipo de producto (factoring/leasing).
            estado: Estado de la evaluación.
            puntaje: Puntaje de riesgo.
            ratios_json: JSON de ratios financieros.
            politicas_json: JSON de políticas aplicadas.
            dictamen: Texto del dictamen.
            condiciones: Lista de condiciones separadas por |.
            factores_positivos: Lista de factores positivos separados por |.
            factores_riesgo: Lista de factores de riesgo separados por |.
            recomendaciones: Lista de recomendaciones separadas por |.

        Returns:
            JSON del resultado estructurado.
        """
        resultado = ResultadoEvaluacion(
            rut_empresa=rut,
            producto=TipoProducto(producto),
            estado=EstadoEvaluacion(estado),
            puntaje_riesgo=puntaje,
            ratios_calculados=RatiosFinancieros.model_validate_json(ratios_json),
            politicas_aplicadas=PoliticasEvaluacion.model_validate_json(politicas_json),
            dictamen=dictamen,
            condiciones_aplicables=condiciones.split("|") if condiciones else [],
            factores_positivos=factores_positivos.split("|") if factores_positivos else [],
            factores_riesgo=factores_riesgo.split("|") if factores_riesgo else [],
            recomendaciones=recomendaciones.split("|") if recomendaciones else [],
        )

        return resultado.model_dump_json(indent=2)

    def _crear_agente(self):
        """Crea el agente LangChain con prompt especializado.

        Returns:
            AgentExecutor configurado.
        """
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Eres un Dictaminador Crediticio Senior, experto en informes de evaluación financiera.\n\n"
                "Tu función es GENERAR EL DICTAMEN FINAL integrando todos los análisis previos:\n"
                "1. Datos de la empresa (del Ingestor)\n"
                "2. Ratios financieros (del Analista)\n"
                "3. Evaluación de riesgo (del Evaluador)\n\n"
                "ESTRUCTURA DEL DICTAMEN:\n"
                "1. RESUMEN EJECUTIVO: Estado de la evaluación y puntaje de riesgo\n"
                "2. ANTECEDENTES DE LA EMPRESA: RUT, razón social, antigüedad, actividad\n"
                "3. ANÁLISIS FINANCIERO: Principales ratios y su interpretación\n"
                "4. EVALUACIÓN DE RIESGO: Incumplimientos detectados y factores considerados\n"
                "5. CONCLUSIONES: Decisión final con condiciones aplicables\n"
                "6. RECOMENDACIONES: Acciones sugeridas al solicitante\n\n"
                "REGLAS:\n"
                "- Usa un tono profesional y formal\n"
                "- Sé específico con cifras y ratios\n"
                "- Fundamenta cada conclusión\n"
                "- Usa la herramienta estructurar_resultado para generar el output final\n"
                "- Separa listas con el caracter '|'"
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

    def generar_dictamen(
        self,
        info_tributaria: InfoTributaria,
        estados_financieros: EstadosFinancieros,
        info_crediticia: InfoCrediticia,
        ratios: RatiosFinancieros,
        politicas: PoliticasEvaluacion,
        incumplimientos: List[str],
        puntaje_riesgo: float,
        estado: EstadoEvaluacion,
    ) -> Dict[str, Any]:
        """Genera el dictamen crediticio completo.

        Args:
            info_tributaria: Información tributaria de la empresa.
            estados_financieros: Estados financieros.
            info_crediticia: Información crediticia.
            ratios: Ratios financieros calculados.
            politicas: Políticas aplicadas.
            incumplimientos: Lista de incumplimientos detectados.
            puntaje_riesgo: Puntaje de riesgo calculado.
            estado: Estado de la evaluación.

        Returns:
            Diccionario con el resultado completo de la evaluación.
        """
        input_text = (
            f"Genera el dictamen crediticio final con los siguientes datos:\n\n"
            f"--- DATOS DE LA EMPRESA ---\n"
            f"Info Tributaria: {info_tributaria.model_dump_json()}\n"
            f"Estados Financieros: {estados_financieros.model_dump_json()}\n"
            f"Info Crediticia: {info_crediticia.model_dump_json()}\n\n"
            f"--- ANÁLISIS FINANCIERO ---\n"
            f"Ratios: {ratios.model_dump_json()}\n\n"
            f"--- EVALUACIÓN DE RIESGO ---\n"
            f"Políticas: {politicas.model_dump_json()}\n"
            f"Incumplimientos: {incumplimientos}\n"
            f"Puntaje de Riesgo: {puntaje_riesgo:.1f}/100\n"
            f"Estado: {estado.value}\n\n"
            f"Genera el dictamen completo y estructura el resultado final."
        )

        resultado = self.agent.invoke({"input": input_text})
        return resultado