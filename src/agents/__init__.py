"""Agentes especializados del sistema de evaluación crediticia."""

from src.agents.ingestor_agent import IngestorAgent
from src.agents.analista_financiero import AnalistaFinancieroAgent
from src.agents.evaluador_riesgo import EvaluadorRiesgoAgent
from src.agents.dictaminador import DictaminadorAgent

__all__ = [
    "IngestorAgent",
    "AnalistaFinancieroAgent",
    "EvaluadorRiesgoAgent",
    "DictaminadorAgent",
]
