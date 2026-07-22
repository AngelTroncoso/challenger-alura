"""Agente Evaluador de Riesgo: especializado en aplicar políticas crediticias."""

from typing import Any, Dict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseLanguageModel
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.tools import Tool
from src.tools.policy_engine_tool import evaluar_cumplimiento, determinar_estado
from src.models.schemas import (
    RatiosFinancieros,
    PoliticasEvaluacion,
    InfoCrediticia,
    InfoTributaria,
    EstadoEvaluacion,
)


class EvaluadorRiesgoAgent:
    """Agente especializado en evaluación de riesgo crediticio.

    Este agente aplica las políticas CMF e internas configuradas
    contra los ratios financieros calculados y la información crediticia,
    determinando el estado de la evaluación y generando un scoring.
    """

    def __init__(self, llm: BaseLanguageModel):
        """Inicializa el agente con un modelo de lenguaje.

        Args:
            llm: Modelo de lenguaje para interpretación de resultados.
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
                name="evaluar_cumplimiento_politicas",
                func=self._evaluar_wrapper,
                description=(
                    "Evalúa el cumplimiento de políticas crediticias. "
                    "Requiere: ratios_financieros (JSON), politicas (JSON), "
                    "info_crediticia (JSON), info_tributaria (JSON). "
                    "Retorna lista de incumplimientos y puntaje de riesgo."
                ),
            ),
            Tool(
                name="determinar_estado_evaluacion",
                func=lambda inc_list, puntaje: str(
                    determinar_estado(
                        inc_list if isinstance(inc_list, list) else eval(inc_list),
                        float(puntaje),
                    )
                ),
                description=(
                    "Determina el estado final de la evaluación "
                    "(aprobado, aprobado con condiciones, rechazado, requiere análisis manual). "
                    "Input: lista de incumplimientos y puntaje de riesgo."
                ),
            ),
        ]

    def _evaluar_wrapper(self, ratios_json: str, politicas_json: str,
                         info_cred_json: str, info_trib_json: str) -> str:
        """Wrapper para evaluar cumplimiento desde strings JSON.

        Args:
            ratios_json: JSON de RatiosFinancieros.
            politicas_json: JSON de PoliticasEvaluacion.
            info_cred_json: JSON de InfoCrediticia.
            info_trib_json: JSON de InfoTributaria.

        Returns:
            String con resultados de la evaluación.
        """
        ratios = RatiosFinancieros.model_validate_json(ratios_json)
        politicas = PoliticasEvaluacion.model_validate_json(politicas_json)
        info_cred = InfoCrediticia.model_validate_json(info_cred_json)
        info_trib = InfoTributaria.model_validate_json(info_trib_json)

        incumplimientos, puntaje = evaluar_cumplimiento(
            ratios, politicas, info_cred, info_trib
        )

        estado = determinar_estado(incumplimientos, puntaje)

        return (
            f"Puntaje de riesgo: {puntaje:.1f}/100\n"
            f"Estado: {estado.value}\n"
            f"Incumplimientos ({len(incumplimientos)}):\n" +
            "\n".join(f"  - {i}" for i in incumplimientos)
        )

    def _crear_agente(self):
        """Crea el agente LangChain con prompt especializado.

        Returns:
            AgentExecutor configurado.
        """
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Eres un Evaluador de Riesgo Crediticio experto en normativa CMF chilena.\n\n"
                "Tu función es:\n"
                "1. Recibir los ratios financieros calculados por el Analista\n"
                "2. Aplicar las políticas de evaluación configuradas\n"
                "3. Identificar incumplimientos y calcular puntaje de riesgo\n"
                "4. Determinar el estado final de la evaluación\n\n"
                "CRITERIOS CMF:\n"
                "- Liquidez mínima: 1.2 (índice de liquidez corriente)\n"
                "- Endeudamiento máximo: 2.5 (pasivo total / activo total)\n"
                "- Cobertura EBITDA mínima: 1.5 (para leasing)\n"
                "- Antigüedad mínima: 12 meses de operación\n"
                "- Protestos y morosidades: factores de rechazo\n\n"
                "REGLAS:\n"
                "- Usa evaluar_cumplimiento_politicas para la validación técnica\n"
                "- Usa determinar_estado_evaluacion para la clasificación final\n"
                "- Proporciona una explicación detallada de cada incumplimiento\n"
                "- Identifica factores de riesgo y factores mitigantes"
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

    def evaluar(
        self,
        ratios: RatiosFinancieros,
        politicas: PoliticasEvaluacion,
        info_crediticia: InfoCrediticia,
        info_tributaria: InfoTributaria,
    ) -> Dict[str, Any]:
        """Ejecuta la evaluación de riesgo crediticio.

        Args:
            ratios: Ratios financieros calculados.
            politicas: Políticas de evaluación configuradas.
            info_crediticia: Información crediticia del deudor.
            info_tributaria: Información tributaria de la empresa.

        Returns:
            Diccionario con resultados de la evaluación.
        """
        input_text = (
            f"Evalúa el riesgo crediticio con los siguientes datos:\n\n"
            f"Ratios Financieros: {ratios.model_dump_json()}\n"
            f"Políticas Aplicadas: {politicas.model_dump_json()}\n"
            f"Info Crediticia: {info_crediticia.model_dump_json()}\n"
            f"Info Tributaria: {info_tributaria.model_dump_json()}\n\n"
            "Aplica las políticas, identifica incumplimientos, "
            "calcula el puntaje de riesgo y determina el estado final."
        )

        resultado = self.agent.invoke({"input": input_text})
        return resultado