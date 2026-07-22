"""Validadores para RUT chileno y formatos numéricos/tributarios chilenos."""

import re
from typing import Union


# ---------------------------------------------------------------------------
# RUT chileno (módulo 11)
# ---------------------------------------------------------------------------

def limpiar_rut(rut: str) -> str:
    """Limpia un RUT eliminando puntos, guiones y espacios.

    Args:
        rut: RUT en cualquier formato (ej: ``12.345.678-5``, ``12345678-5``).

    Returns:
        RUT limpio sin puntos ni guion (ej: ``123456785``).
    """
    return re.sub(r"[\.\-\s]", "", rut.strip())


def formatear_rut(rut: str) -> str:
    """Formatea un RUT al formato estándar chileno: ``XX.XXX.XXX-X``.

    Args:
        rut: RUT limpio o con formato.

    Returns:
        RUT formateado con puntos y guion.
    """
    limpio = limpiar_rut(rut)
    if len(limpio) < 2:
        return rut
    cuerpo = limpio[:-1]
    dv = limpio[-1].upper()
    # Agregar puntos cada 3 dígitos desde la derecha
    cuerpo_formateado = "{:,}".format(int(cuerpo)).replace(",", ".")
    return f"{cuerpo_formateado}-{dv}"


def _calcular_dv(rut_sin_dv: int) -> str:
    """Calcula el dígito verificador (módulo 11) para un RUT.

    Args:
        rut_sin_dv: Parte numérica del RUT sin dígito verificador.

    Returns:
        Dígito verificador como string (``0``-``9`` o ``K``).
    """
    suma = 0
    multiplicador = 2
    while rut_sin_dv > 0:
        suma += (rut_sin_dv % 10) * multiplicador
        rut_sin_dv //= 10
        multiplicador += 1
        if multiplicador > 7:
            multiplicador = 2
    resto = suma % 11
    dv_calculado = 11 - resto
    if dv_calculado == 11:
        return "0"
    elif dv_calculado == 10:
        return "K"
    else:
        return str(dv_calculado)


def validar_rut(rut: str) -> bool:
    """Valida un RUT chileno usando el algoritmo del módulo 11.

    Acepta RUT con o sin puntos, con o sin guion, y dígito verificador
    ``K`` mayúscula o minúscula.

    Args:
        rut: RUT a validar.

    Returns:
        ``True`` si el RUT es válido, ``False`` en caso contrario.
    """
    try:
        limpio = limpiar_rut(rut)
        if len(limpio) < 2:
            return False
        cuerpo_str = limpio[:-1]
        dv_ingresado = limpio[-1].upper()
        if not cuerpo_str.isdigit():
            return False
        cuerpo = int(cuerpo_str)
        dv_calculado = _calcular_dv(cuerpo)
        return dv_calculado == dv_ingresado
    except (ValueError, IndexError):
        return False


# ---------------------------------------------------------------------------
# Parseo de montos / números en formato chileno
# ---------------------------------------------------------------------------

def parsear_monto(valor: Union[str, int, float]) -> float:
    """Convierte un string con formato numérico chileno a ``float``.

    El formato chileno usa punto (``.``) como separador de miles y
    coma (``,``) como separador decimal.

    Ejemplos::

        >>> parsear_monto("1.234.567,89")
        1234567.89
        >>> parsear_monto("$ 1.234.567,89")
        1234567.89
        >>> parsear_monto("1234.56")  # formato inglés
        1234.56
        >>> parsear_monto(1234.56)
        1234.56

    Args:
        valor: String, int o float con el valor a parsear.

    Returns:
        Número como ``float``.
    """
    if isinstance(valor, (int, float)):
        return float(valor)

    if not isinstance(valor, str):
        return 0.0

    # Limpiar: eliminar símbolo $, espacios, UF, etc.
    texto = valor.strip()
    texto = re.sub(r"[$\s]", "", texto)

    # Detectar si usa formato chileno (coma decimal) o inglés (punto decimal)
    # Formato chileno: "1.234.567,89" -> tiene puntos de miles y coma decimal
    # Formato inglés: "1234567.89" -> solo un punto o ninguno
    if "," in texto:
        # Formato chileno: eliminar puntos (miles), reemplazar coma por punto
        texto = texto.replace(".", "")
        texto = texto.replace(",", ".")
    else:
        # Formato inglés o sin separadores: solo eliminar posibles puntos de miles
        # Si hay más de un punto, es formato con puntos de miles
        puntos = texto.count(".")
        if puntos > 1:
            texto = texto.replace(".", "")

    try:
        return float(texto)
    except ValueError:
        return 0.0


def formatear_monto(valor: float, incluir_simbolo: bool = True) -> str:
    """Formatea un número al formato chileno con separadores.

    Ejemplos::

        >>> formatear_monto(1234567.89)
        '$ 1.234.567,89'
        >>> formatear_monto(1234567.89, incluir_simbolo=False)
        '1.234.567,89'

    Args:
        valor: Número a formatear.
        incluir_simbolo: Si se incluye el símbolo ``$``.

    Returns:
        String con el monto formateado.
    """
    entero = int(valor)
    decimales = round(abs(valor) - abs(entero), 2)
    # Formatear parte entera con puntos de miles
    entero_str = "{:,}".format(entero).replace(",", ".")
    # Parte decimal (2 dígitos)
    decimal_str = f"{decimales:.2f}".split(".")[1]
    resultado = f"{entero_str},{decimal_str}"
    if incluir_simbolo:
        resultado = f"$ {resultado}"
    return resultado


def parsear_porcentaje(valor: Union[str, int, float]) -> float:
    """Convierte un string con formato de porcentaje a float (0-1).

    Ejemplos::

        >>> parsear_porcentaje("15%")
        0.15
        >>> parsear_porcentaje("12,5%")
        0.125
        >>> parsear_porcentaje(0.15)
        0.15

    Args:
        valor: Porcentaje como string (``"15%"``) o float (``0.15``).

    Returns:
        Valor decimal entre 0 y 1.
    """
    if isinstance(valor, (int, float)):
        # Si es > 1, asumir que es porcentaje (ej: 15 -> 0.15)
        if valor > 1:
            return float(valor) / 100.0
        return float(valor)

    texto = str(valor).strip().replace("%", "").replace(" ", "")
    numero = parsear_monto(texto)
    # Si el número es > 1, asumir que es porcentaje (ej: 15 -> 0.15)
    if numero > 1:
        return numero / 100.0
    return numero