"""Text utilities - 文本处理工具函数

通用的文本转换和处理工具，不依赖特定业务逻辑。
"""

from __future__ import annotations

import inflection


def to_snake_case(name: str) -> str:
    """任何格式 -> snake_case (下划线)

    将各种命名风格转换为下划线分隔的小写格式。
    完美处理连续大写字母（如 PDF -> pdf）。

    Args:
        name: 输入名称（支持 kebab-case, camelCase, PascalCase）

    Returns:
        snake_case 格式的字符串

    Examples:
        >>> to_snake_case("PDFProcessing")
        'pdf_processing'
        >>> to_snake_case("get-weather-mock")
        'get_weather_mock'
        >>> to_snake_case("getWeatherMock")
        'get_weather_mock'
        >>> to_snake_case("GetWeatherMock")
        'get_weather_mock'
    """
    return inflection.underscore(name)


def to_kebab_case(name: str) -> str:
    """任何格式 -> kebab-case (连字符)

    将各种命名风格转换为连字符分隔的小写格式。
    完美处理连续大写字母（如 PDF -> pdf）。

    Args:
        name: 输入名称（支持 snake_case, camelCase, PascalCase）

    Returns:
        kebab-case 格式的字符串

    Examples:
        >>> to_kebab_case("PDFProcessing")
        'pdf-processing'
        >>> to_kebab_case("get_weather_mock")
        'get-weather-mock'
        >>> to_kebab_case("getWeatherMock")
        'get-weather-mock'
        >>> to_kebab_case("GetWeatherMock")
        'get-weather-mock'
    """
    return inflection.dasherize(inflection.underscore(name))


def to_camel_case(name: str) -> str:
    """任何格式 -> camelCase (小驼峰)

    将各种命名风格转换为驼峰格式（首字母小写）。

    Args:
        name: 输入名称（支持 kebab-case, snake_case, PascalCase）

    Returns:
        camelCase 格式的字符串

    Examples:
        >>> to_camel_case("pdf-processing")
        'pdfProcessing'
        >>> to_camel_case("pdf_processing")
        'pdfProcessing'
        >>> to_camel_case("PDFProcessing")
        'pdfProcessing'
    """
    return inflection.camelize(to_snake_case(name), uppercase_first_letter=False)


def to_pascal_case(name: str) -> str:
    """任何格式 -> PascalCase (大驼峰/帕斯卡)

    将各种命名风格转换为帕斯卡格式（首字母大写）。

    Args:
        name: 输入名称（支持 kebab-case, snake_case, camelCase）

    Returns:
        PascalCase 格式的字符串

    Examples:
        >>> to_pascal_case("pdf-processing")
        'PdfProcessing'
        >>> to_pascal_case("pdf_processing")
        'PdfProcessing'
        >>> to_pascal_case("pdfProcessing")
        'PdfProcessing'
    """
    return inflection.camelize(to_snake_case(name), uppercase_first_letter=True)


def to_title(kebab_name: str) -> str:
    """任何格式 -> Title Case (标题格式)

    将各种命名风格转换为标题格式（每个单词首字母大写）。

    Args:
        kebab_name: 输入名称（支持 kebab-case, snake_case, camelCase）

    Returns:
        Title Case 格式的字符串

    Examples:
        >>> to_title("pdf-processing")
        'Pdf Processing'
        >>> to_title("pdf_processing")
        'Pdf Processing'
        >>> to_title("get-weather")
        'Get Weather'
        >>> to_title("PDFProcessing")
        'Pdf Processing'
    """
    return inflection.titleize(to_snake_case(kebab_name))
