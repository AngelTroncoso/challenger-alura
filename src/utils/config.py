"""Carga de configuración y variables de entorno para el sistema."""

import os
from typing import Optional

from dotenv import load_dotenv

# Cargar variables de entorno desde .env una sola vez al importar
load_dotenv()


# ---------------------------------------------------------------------------
# Groq API
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "llama-3.3-70b-versatile"
"""Modelo por defecto para inferencia con Groq."""

# Modelos alternativos disponibles:
# - "deepseek-r1-distill-llama-70b"
# - "mixtral-8x7b-32768"
# - "llama-3.1-8b-instant"


def get_groq_api_key() -> str:
    """Retorna la API Key de Groq desde variables de entorno.

    Raises:
        ValueError: Si la variable ``GROQ_API_KEY`` no está configurada.

    Returns:
        La API Key como string.
    """
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError(
            "GROQ_API_KEY no está configurada. "
            "Créala en un archivo .env o configúrala como variable de entorno."
        )
    return key


def get_model_name() -> str:
    """Retorna el nombre del modelo LLM configurado.

    La variable de entorno ``MODEL_NAME`` permite sobreescribir el valor
    por defecto.

    Returns:
        Nombre del modelo (ej: ``"llama-3.3-70b-versatile"``).
    """
    return os.getenv("MODEL_NAME", DEFAULT_MODEL)


def get_temperature() -> float:
    """Retorna la temperatura para el LLM.

    La variable de entorno ``LLM_TEMPERATURE`` permite configurarla.

    Returns:
        Valor de temperatura (default ``0.1`` para tareas analíticas).
    """
    try:
        return float(os.getenv("LLM_TEMPERATURE", "0.1"))
    except (TypeError, ValueError):
        return 0.1


def get_max_tokens() -> int:
    """Retorna el máximo de tokens de salida para el LLM.

    Returns:
        Máximo de tokens (default ``4096``).
    """
    try:
        return int(os.getenv("LLM_MAX_TOKENS", "4096"))
    except (TypeError, ValueError):
        return 4096


# ---------------------------------------------------------------------------
# Streamlit Cloud helpers
# ---------------------------------------------------------------------------

def is_streamlit_cloud() -> bool:
    """Detecta si la aplicación se está ejecutando en Streamlit Cloud.

    Streamlit Cloud define la variable de entorno ``SERVER_SOFTWARE``.
    """
    server_software = os.getenv("SERVER_SOFTWARE", "")
    return "streamlit" in server_software.lower()


def get_groq_api_key_safe() -> Optional[str]:
    """Retorna la API Key de Groq sin lanzar excepción.

    Útil para interfaces donde se quiere mostrar un mensaje amigable
    si la clave no está configurada, en lugar de una traza de error.

    Returns:
        La API Key o ``None`` si no está configurada.
    """
    try:
        return get_groq_api_key()
    except ValueError:
        return None