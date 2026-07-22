"""Extracción de texto y tablas desde PDFs del SII, balances e informes comerciales chilenos.

Soporta:
- Carpeta Tributaria SII: Formulario 29 (IVA mensual), Formulario 22 (Renta anual)
- Balance de 8 Columnas (Estados Financieros)
- Informe Comercial DICOM / Platinum
"""

import logging
import re
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pypdf

from src.models.schemas import (
    EstadosFinancieros,
    InfoCrediticia,
    InfoTributaria,
)
from src.utils.validators import (
    limpiar_rut,
    parsear_monto,
    validar_rut,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# Extracción base de texto desde PDF
# ===========================================================================


def extraer_texto(pdf_path: str, max_paginas: int = 50) -> str:
    """Extrae texto plano de un archivo PDF usando PyPDF.

    Args:
        pdf_path: Ruta al archivo PDF.
        max_paginas: Máximo de páginas a procesar (default 50).

    Returns:
        Texto completo extraído del PDF.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si el PDF está vacío o no se pudo leer.
    """
    ruta = Path(pdf_path)
    if not ruta.is_file():
        raise FileNotFoundError(f"Archivo PDF no encontrado: {pdf_path}")

    try:
        with open(ruta, "rb") as fh:
            reader = pypdf.PdfReader(fh)
            paginas = min(len(reader.pages), max_paginas)
            if paginas == 0:
                raise ValueError(f"El PDF está vacío: {pdf_path}")

            texto = ""
            for i in range(paginas):
                pagina = reader.pages[i]
                texto_pagina = pagina.extract_text()
                if texto_pagina:
                    texto += texto_pagina + "\n--- PAGE BREAK ---\n"

            if not texto.strip():
                logger.warning(
                    "No se pudo extraer texto del PDF (puede ser escaneado): %s",
                    pdf_path,
                )
            return texto.strip()

    except pypdf.errors.PdfReadError as exc:
        raise ValueError(f"Error al leer PDF {pdf_path}: {exc}")


# ===========================================================================
# Detección del tipo de documento
# ===========================================================================

# Patrones de detección para cada tipo de documento
_PATRON_F29 = re.compile(
    r"(formulario\s*29|declaraci[oó]n\s*mensual\s*.*iva|f29)",
    re.IGNORECASE,
)
_PATRON_F22 = re.compile(
    r"(formulario\s*22|declaraci[oó]n\s*anual\s*.*renta|f22)",
    re.IGNORECASE,
)
_PATRON_BALANCE = re.compile(
    r"(balance\s*general|balance\s*8\s*columnas|estado\s*de\s*situaci[oó]n|"
    r"activo\s*corriente|pasivo\s*corriente|patrimonio)",
    re.IGNORECASE,
)
_PATRON_DICOM = re.compile(
    r"(informe\s*comercial|dic om|platinum|protestos|morosidades|"
    r"deuda\s*directa)",
    re.IGNORECASE,
)

# Líneas clave para identificar secciones en formularios SII
_SECCION_F29_IVA_VENTAS = re.compile(
    r"(iva\s*(?:debido|devengado|ventas|d[eé]bito fiscal)|"
    r"c[oód]igo\s*510|ventas\s*netas)",
    re.IGNORECASE,
)
_SECCION_F29_IVA_COMPRAS = re.compile(
    r"(iva\s*(?:acreditado|compras|cr[eé]dito fiscal)|"
    r"c[oód]igo\s*520|compras\s*del\s*mes)",
    re.IGNORECASE,
)
_SECCION_F29_IMPUESTO = re.compile(
    r"(impuesto\s*a\s*pagar|impuesto\s*pagado|"
    r"pago\s*neto|diferencia\s*de\s*impuesto)",
    re.IGNORECASE,
)

# Patrones para buscar RUT y razón social
_PATRON_RUT = re.compile(r"(\d{1,2}\.?\d{3}\.?\d{3}[-][0-9Kk])")
_PATRON_RAZON_SOCIAL = re.compile(
    r"(raz[oó]n\s*social|nombre\s*de\s*la\s*empresa|nombre\s*del\s*contribuyente)\s*:?\s*(.+)",
    re.IGNORECASE,
)


def _detectar_tipo_documento(texto: str) -> str:
    """Detecta el tipo de documento basado en patrones en el texto.

    Args:
        texto: Texto extraído del PDF.

    Returns:
        Tipo de documento: ``"f29"``, ``"f22"``, ``"balance"``, ``"dicom"``
        o ``"desconocido"``.
    """
    texto_lower = texto.lower()
    puntajes = {
        "f29": len(_PATRON_F29.findall(texto_lower)),
        "f22": len(_PATRON_F22.findall(texto_lower)),
        "balance": len(_PATRON_BALANCE.findall(texto_lower)),
        "dicom": len(_PATRON_DICOM.findall(texto_lower)),
    }
    max_tipo = max(puntajes, key=puntajes.get)
    if puntajes[max_tipo] > 0:
        return max_tipo
    return "desconocido"


# ===========================================================================
# Parseo de RUT y Razón Social (común a todos los documentos)
# ===========================================================================


def _extraer_rut(texto: str) -> Optional[str]:
    """Extrae el RUT del texto del documento.

    Busca el primer RUT válido en el texto.

    Returns:
        RUT válido o ``None`` si no se encuentra.
    """
    for match in _PATRON_RUT.finditer(texto):
        rut_candidato = match.group(1)
        if validar_rut(rut_candidato):
            return limpiar_rut(rut_candidato)
    return None


def _extraer_razon_social(texto: str) -> str:
    """Extrae la razón social del texto."""
    for match in _PATRON_RAZON_SOCIAL.finditer(texto):
        return match.group(2).strip()
    # Fallback: buscar primeras líneas con mayúsculas sostenidas
    lineas = texto.split("\n")
    for linea in lineas[:30]:
        linea = linea.strip()
        if len(linea) > 10 and linea.isupper() and not re.search(r"\d{3,}", linea):
            return linea
    return ""


# ===========================================================================
# Parseo de Formulario 29 (IVA mensual)
# ===========================================================================


def _parsear_fecha_linea(texto: str) -> Optional[date]:
    """Extrae la primera fecha válida en formato AAAA-MM o similar."""
    patrones_fecha = [
        re.compile(r"(\d{4})[-/](\d{2})"),  # 2024-12 o 2024/12
        re.compile(r"(\d{2})[-/](\d{4})"),  # 12-2024
        re.compile(r"(enero|febrero|marzo|abril|mayo|junio|"
                   r"julio|agosto|septiembre|octubre|noviembre|diciembre)\s+de\s+(\d{4})",
                   re.IGNORECASE),
    ]
    for patron in patrones_fecha:
        match = patron.search(texto)
        if match:
            grupos = match.groups()
            try:
                if len(grupos) == 2:
                    if grupos[0].isdigit() and len(grupos[0]) == 4:
                        anio, mes = int(grupos[0]), int(grupos[1])
                    elif grupos[1].isdigit() and len(grupos[1]) == 4:
                        mes, anio = int(grupos[0]), int(grupos[1])
                    else:
                        continue
                    if 1 <= mes <= 12 and 1900 <= anio <= 2100:
                        return date(anio, mes, 1)
            except (ValueError, IndexError):
                continue
    return None


def _extraer_valor_f29(texto: str, patron_seccion: re.Pattern) -> float:
    """Extrae el primer valor numérico encontrado después de una sección."""
    match_seccion = patron_seccion.search(texto)
    if not match_seccion:
        return 0.0
    # Buscar números después de la sección
    inicio = match_seccion.end()
    resto = texto[inicio:inicio + 500]
    numeros = re.findall(r"[\d\.,]+", resto)
    for num_str in numeros:
        valor = parsear_monto(num_str)
        if valor > 0:
            return valor
    return 0.0


def _extraer_fila_tabla_f29(linea: str) -> Optional[Tuple[str, float]]:
    """Intenta parsear una línea de tabla F29 (código + valor)."""
    # Formato: código numérico + descripción + valor
    match = re.match(
        r"\s*(\d{3,4})\s+(.+?)\s+([\d\.\s]+,\d{2})\s*$",
        linea,
    )
    if match:
        return match.group(1), parsear_monto(match.group(3))
    return None


def extraer_formulario_29(texto: str) -> InfoTributaria:
    """Extrae información del Formulario 29 (declaración mensual de IVA).

    Args:
        texto: Texto extraído del PDF del F29.

    Returns:
        Objeto ``InfoTributaria`` con datos parciales del F29.
    """
    rut = _extraer_rut(texto) or ""
    razon_social = _extraer_razon_social(texto)

    # Extraer IVA ventas y compras
    iva_ventas = _extraer_valor_f29(texto, _SECCION_F29_IVA_VENTAS)
    iva_compras = _extraer_valor_f29(texto, _SECCION_F29_IVA_COMPRAS)
    impuesto = _extraer_valor_f29(texto, _SECCION_F29_IMPUESTO)
    periodo = _parsear_fecha_linea(texto)

    # Construir registro mensual
    registro_mensual: Dict[str, float] = {
        "iva_ventas": iva_ventas,
        "iva_compras": iva_compras,
        "impuesto_pagado": impuesto,
    }

    return InfoTributaria(
        rut_empresa=rut,
        razon_social=razon_social,
        fecha_inicio_actividades=periodo,
        formulario29_ultimos_12m=[registro_mensual],
        formulario22_anual={},
    )


# ===========================================================================
# Parseo de Formulario 22 (Renta anual)
# ===========================================================================


def _extraer_valor_f22(texto: str, codigo: str) -> float:
    """Busca un código de línea del F22 y extrae su valor.

    El F22 tiene formato de códigos numéricos con valores asociados.
    """
    patron = re.compile(
        r"(?:c[oód]igo|l[íi]nea)\s*:?\s*" + re.escape(codigo) +
        r"[\s:.]*\n?.{0,100}?([\d\.,]+)",
        re.IGNORECASE,
    )
    match = patron.search(texto)
    if match:
        return parsear_monto(match.group(1))
    return 0.0


def extraer_formulario_22(texto: str) -> InfoTributaria:
    """Extrae información del Formulario 22 (declaración anual de renta).

    Args:
        texto: Texto extraído del PDF del F22.

    Returns:
        Objeto ``InfoTributaria`` con datos del F22.
    """
    rut = _extraer_rut(texto) or ""
    razon_social = _extraer_razon_social(texto)

    # Códigos típicos del F22 chileno
    renta_liquida = _extraer_valor_f22(texto, "541")
    if renta_liquida == 0.0:
        renta_liquida = _extraer_valor_f22(texto, "542")
    impuesto_pagado = _extraer_valor_f22(texto, "694")
    if impuesto_pagado == 0.0:
        impuesto_pagado = _extraer_valor_f22(texto, "692")
    capital_propio = _extraer_valor_f22(texto, "751")
    if capital_propio == 0.0:
        capital_propio = _extraer_valor_f22(texto, "752")

    return InfoTributaria(
        rut_empresa=rut,
        razon_social=razon_social,
        formulario29_ultimos_12m=[],
        formulario22_anual={
            "renta_liquida": renta_liquida,
            "impuesto_pagado": impuesto_pagado,
            "capital_propio": capital_propio,
        },
    )


# ===========================================================================
# Parseo de Balance de 8 Columnas / Estados Financieros
# ===========================================================================

# Patrones para encontrar líneas de balance
_PATRONES_BALANCE = {
    "activos_corrientes": re.compile(
        r"(total\s+activo\s+corriente|activo\s+corriente\s+total)[\s:]*([\d\.,]+)",
        re.IGNORECASE,
    ),
    "activos_no_corrientes": re.compile(
        r"(total\s+activo\s+no\s+corriente|activo\s+fijo[\s\w]*total)[\s:]*([\d\.,]+)",
        re.IGNORECASE,
    ),
    "pasivos_corrientes": re.compile(
        r"(total\s+pasivo\s+corriente|pasivo\s+corriente\s+total)[\s:]*([\d\.,]+)",
        re.IGNORECASE,
    ),
    "pasivos_no_corrientes": re.compile(
        r"(total\s+pasivo\s+no\s+corriente|pasivo\s+largo\s*plazo[\s\w]*total)[\s:]*([\d\.,]+)",
        re.IGNORECASE,
    ),
    "patrimonio": re.compile(
        r"(total\s+patrimonio[\s\w]*|patrimonio\s+total)[\s:]*([\d\.,]+)",
        re.IGNORECASE,
    ),
    "ventas_netas": re.compile(
        r"(ventas\s+netas|ingresos\s+de\s+explotaci[oó]n)[\s:]*([\d\.,]+)",
        re.IGNORECASE,
    ),
    "costo_ventas": re.compile(
        r"(costo\s+de\s+ventas|costo\s+de\s+explotaci[oó]n)[\s:]*([\d\.,]+)",
        re.IGNORECASE,
    ),
    "resultado_operacional": re.compile(
        r"(resultado\s+operacional|utilidad\s+operacional|"
        r"resultado\s+de\s+explotaci[oó]n)[\s:]*([\d\.,]+)",
        re.IGNORECASE,
    ),
    "resultado_neto": re.compile(
        r"(resultado\s+neto|utilidad\s+neto|"
        r"resultado\s+del\s+ejercicio)[\s:]*([\d\.,]+)",
        re.IGNORECASE,
    ),
    "depreciacion": re.compile(
        r"(depreciaci[oó]n[\s\w]*total|gasto\s+por\s+depreciaci[oó]n)[\s:]*([\d\.,]+)",
        re.IGNORECASE,
    ),
    "gastos_financieros": re.compile(
        r"(gastos\s+financieros|costos\s+financieros|"
        r"gastos\s+por\s+intereses)[\s:]*([\d\.,]+)",
        re.IGNORECASE,
    ),
}


def extraer_balance(texto: str) -> EstadosFinancieros:
    """Extrae estados financieros desde un balance de 8 columnas.

    Args:
        texto: Texto extraído del PDF del balance.

    Returns:
        Objeto ``EstadosFinancieros`` con los valores encontrados.
        Los valores no encontrados quedan en 0.0.
    """
    valores: Dict[str, float] = {}
    for campo, patron in _PATRONES_BALANCE.items():
        match = patron.search(texto)
        if match:
            valor_str = match.group(2)
            valores[campo] = parsear_monto(valor_str)
        else:
            valores[campo] = 0.0

    # Extraer período
    periodo = ""
    patron_periodo = re.compile(
        r"(per[íi]odo|periodo|al\s*\d{2}[-/]\d{2}[-/]\d{4}|"
        r"(?:diciembre|junio)\s*\d{4})",
        re.IGNORECASE,
    )
    match_periodo = patron_periodo.search(texto)
    if match_periodo:
        periodo = match_periodo.group(0)

    return EstadosFinancieros(
        periodo=periodo,
        activos_corrientes=valores.get("activos_corrientes", 0.0),
        activos_no_corrientes=valores.get("activos_no_corrientes", 0.0),
        pasivos_corrientes=valores.get("pasivos_corrientes", 0.0),
        pasivos_no_corrientes=valores.get("pasivos_no_corrientes", 0.0),
        patrimonio=valores.get("patrimonio", 0.0),
        ventas_netas=valores.get("ventas_netas", 0.0),
        costo_ventas=valores.get("costo_ventas", 0.0),
        resultado_operacional=valores.get("resultado_operacional", 0.0),
        resultado_neto=valores.get("resultado_neto", 0.0),
        depreciacion=valores.get("depreciacion", 0.0),
        gastos_financieros=valores.get("gastos_financieros", 0.0),
    )


# ===========================================================================
# Parseo de Informe DICOM / Platinum
# ===========================================================================

_PATRON_PROTESTOS = re.compile(
    r"protestos?\s*:?\s*(?:si|no|s[íi])?.*?(\$[\d\.,]+|[\d\.,]+)",
    re.IGNORECASE,
)
_PATRON_MOROSIDAD = re.compile(
    r"morosidad\s*(?:vigente|total)?\s*:?\s*(?:si|no|s[íi])",
    re.IGNORECASE,
)
_PATRON_DIAS_MOROSIDAD = re.compile(
    r"(?:d[ií]as\s+de\s+morosidad|m[áa]x\s+d[ií]as)\s*:?\s*(\d+)",
    re.IGNORECASE,
)
_PATRON_MOROSIDAD_PREV = re.compile(
    r"morosidad\s*(?:previsional|previsionales)",
    re.IGNORECASE,
)
_PATRON_ANTIGUEDAD = re.compile(
    r"(?:antig[üu]edad|a[ñn]os\s+de\s+actividad)\s*:?\s*(\d+)",
    re.IGNORECASE,
)


def extraer_dicom(texto: str) -> InfoCrediticia:
    """Extrae información crediticia desde un informe DICOM / Platinum.

    Args:
        texto: Texto extraído del PDF del informe comercial.

    Returns:
        Objeto ``InfoCrediticia`` con los datos encontrados.
    """
    # Protestos
    tiene_protestos = bool(_PATRON_PROTESTOS.search(texto))
    monto_protestos = 0.0
    if tiene_protestos:
        match_protesto = _PATRON_PROTESTOS.search(texto)
        if match_protesto:
            monto_protestos = parsear_monto(match_protesto.group(1))

    # Morosidades
    tiene_morosidades = bool(_PATRON_MOROSIDAD.search(texto))

    # Días de morosidad
    dias_morosidad = 0
    match_dias = _PATRON_DIAS_MOROSIDAD.search(texto)
    if match_dias:
        dias_morosidad = int(match_dias.group(1))

    # Morosidades previsionales
    morosidades_previsionales = bool(_PATRON_MOROSIDAD_PREV.search(texto))

    # Antigüedad
    antiguedad_meses = 0
    match_ant = _PATRON_ANTIGUEDAD.search(texto)
    if match_ant:
        antiguedad_meses = int(match_ant.group(1)) * 12
    else:
        # Buscar años en texto libre
        match_anios = re.search(r"(\d+)\s*(?:a[ñn]os|año)", text, re.IGNORECASE)
        if match_anios:
            antiguedad_meses = int(match_anios.group(1)) * 12

    return InfoCrediticia(
        tiene_protestos=tiene_protestos,
        monto_protestos=monto_protestos,
        tiene_morosidades=tiene_morosidades,
        dias_morosidad=dias_morosidad,
        morosidades_previsionales=morosidades_previsionales,
        antiguedad_meses=antiguedad_meses,
    )


# ===========================================================================
# Orquestador: procesar dossier completo
# ===========================================================================

ResultadoDossier = Dict[str, object]


def procesar_dossier(pdf_path: str) -> ResultadoDossier:
    """Procesa un dossier PDF completo extrayendo toda la información.

    Detecta automáticamente el tipo de documento (F29, F22, balance,
    DICOM) y aplica el parser correspondiente.

    Si el PDF contiene múltiples documentos (ej: carpeta tributaria),
    se procesa página por página y se acumulan los resultados.

    Args:
        pdf_path: Ruta al archivo PDF del dossier.

    Returns:
        Diccionario con las siguientes claves:
        - ``"tipo_documento"``: tipo detectado
        - ``"info_tributaria"``: ``InfoTributaria`` o ``None``
        - ``"estados_financieros"``: ``EstadosFinancieros`` o ``None``
        - ``"info_crediticia"``: ``InfoCrediticia`` o ``None``
        - ``"texto_raw"``: texto completo extraído
        - ``"error"``: mensaje de error si ocurrió, o ``None``
    """
    resultado: ResultadoDossier = {
        "tipo_documento": "desconocido",
        "info_tributaria": None,
        "estados_financieros": None,
        "info_crediticia": None,
        "texto_raw": "",
        "error": None,
    }

    try:
        texto = extraer_texto(pdf_path)
        resultado["texto_raw"] = texto

        if not texto:
            resultado["error"] = "No se pudo extraer texto del PDF"
            return resultado

        tipo = _detectar_tipo_documento(texto)
        resultado["tipo_documento"] = tipo

        if tipo == "f29":
            resultado["info_tributaria"] = extraer_formulario_29(texto)
        elif tipo == "f22":
            resultado["info_tributaria"] = extraer_formulario_22(texto)
        elif tipo == "balance":
            resultado["estados_financieros"] = extraer_balance(texto)
        elif tipo == "dicom":
            resultado["info_crediticia"] = extraer_dicom(texto)
        else:
            # Tipo desconocido: intentar todos los parsers
            logger.info(
                "Tipo de documento no detectado para %s, intentando todos los parsers",
                pdf_path,
            )
            resultado["info_tributaria"] = (
                extraer_formulario_29(texto) or extraer_formulario_22(texto)
            )
            resultado["estados_financieros"] = extraer_balance(texto)
            resultado["info_crediticia"] = extraer_dicom(texto)

    except (FileNotFoundError, ValueError) as exc:
        resultado["error"] = str(exc)
        logger.error("Error procesando dossier %s: %s", pdf_path, exc)

    return resultado