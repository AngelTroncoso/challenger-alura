"""Agente Analista Financiero: especializado en cálculo de ratios y métricas."""

from typing import Any, Dict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseLanguageModel
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.tools import Tool
from src.tools.ratio_calculator_tool import (
    calcular_ratios_factoring,
    calcular_ratios_leasing,
    _calcular_ebitda as calcular_ebitda,
)
from src.models.schemas import (
    EstadosFinancieros,
    InfoCrediticia,
    RatiosFinancieros,
    TipoProducto,
)


class AnalistaFinancieroAgent:
    """Agente especializado en análisis financiero y cálculo de ratios.

    Este agente toma los estados financieros extraídos y calcula
    los ratios específicos según el tipo de producto (factoring/leasing),
    utilizando Pandas para los cálculos.
    """

    def __init__(self, llm: BaseLanguageModel):
        """Inicializa el agente con un modelo de lenguaje.

        Args:
            llm: Modelo de lenguaje para interpretación de métricas.
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
                name="calcular_ratios_factoring",
                func=lambda ee_str, info_str: str(
                    calcular_ratios_factoring(
                        EstadosFinancieros.model_validate_json(ee_str),
                        InfoCrediticia.model_validate_json(info_str),
                    )
                ),
                description=(
                    "Calcula ratios financieros para evaluación de factoring. "
                    "Requiere estados financieros (JSON) e info crediticia (JSON). "
                    "Retorna RatiosFinancieros en JSON."
                ),
            ),
            Tool(
                name="calcular_ratios_leasing",
                func=lambda ee_str: str(
                    calcular_ratios_leasing(
                        EstadosFinancieros.model_validate_json(ee_str),
                    )
                ),
                description=(
                    "Calcula ratios financieros para evaluación de leasing. "
                    "Requiere estados financieros (JSON). "
                    "Retorna RatiosFinancieros en JSON."
                ),
            ),
            Tool(
                name="calcular_ebitda",
                func=lambda ee_str: str(
                    calcular_ebitda(
                        EstadosFinancieros.model_validate_json(ee_str),
                    )
                ),
                description=(
                    "Calcula el EBITDA desde estados financieros. "
                    "Requiere estados financieros (JSON). "
                    "Retorna el valor EBITDA."
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
                "Eres un Analista Financiero especializado en evaluación crediticia chilena.\n\n"
                "Tu función es:\n"
                "1. Recibir los estados financieros estructurados del Ingestor\n"
                "2. Identificar el tipo de producto (factoring o leasing)\n"
                "3. Calcular los ratios financieros correspondientes usando las herramientas\n"
                "4. Interpretar los resultados\n\n"
                "FÓRMULAS CHILENAS ESTÁNDAR:\n"
                "- Liquidez corriente = Activo corriente / Pasivo corriente\n"
                "- Liquidez inmediata = (Activo corriente - Inventarios) / Pasivo corriente\n"
                "- Razón deuda/patrimonio = Pasivo total / Patrimonio\n"
                "- Endeudamiento total = Pasivo total / Activo total\n"
                "- Margen EBITDA = EBITDA / Ventas netas\n"
                "- ROI = Resultado neto / Activo total\n"
                "- Rotación cartera = Ventas netas / Cuentas por cobrar (factoring)\n"
                "- Cobertura FCF = (EBITDA - Capex) / Servicio deuda (leasing)\n\n"
                "REGLAS:\n"
                "- Usa calcular_ratios_factoring para productos de factoring\n"
                "- Usa calcular_ratios_leasing para productos de leasing\n"
                "- Valida que los datos de entrada sean consistentes\n"
                "- Si hay datos faltantes, usa estimaciones conservadoras\n"
                "- Reporta cada ratio con su interpretación"
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

    def analizar(
        self,
        estados_financieros: EstadosFinancieros,
        info_crediticia: InfoCrediticia,
        producto: TipoProducto,
    ) -> Dict[str, Any]:
        """Ejecuta el análisis financiero sobre los datos proporcionados.

        Args:
            estados_financieros: Estados financieros de la empresa.
            info_crediticia: Información crediticia del deudor.
            producto: Tipo de producto (factoring o leasing).

        Returns:
            Diccionario con los ratios calculados e interpretación.
        """
        ee_json = estados_financieros.model_dump_json()
        info_json = info_crediticia.model_dump_json()

        if producto == TipoProducto.FACTORING:
            input_text = (
                f"Analiza los estados financieros para FACTORING:\n"
                f"Estados Financieros: {ee_json}\n"
                f"Info Crediticia: {info_json}\n\n"
                "Calcula todos los ratios de factoring y proporciona interpretación."
            )
        else:
            input_text = (
                f"Analiza los estados financieros para LEASING:\n"
                f"Estados Financieros: {ee_json}\n\n"
                "Calcula todos los ratios de leasing y proporciona interpretación."
            )

        resultado = self.agent.invoke({"input": input_text})
        return resultado