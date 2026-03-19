"""
国际化模块 - 与项目 Config 系统集成
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, Optional, Callable, List, Any

# 导入项目配置系统
try:
    from middleware.config import g_config
except ImportError:
    g_config = None


class I18n:
    """
    国际化翻译管理器
    与 g_config 集成，自动读取和保存语言设置
    """

    _instance = None
    _current_locale = None  # 当前语言代码（如 zh_CN）
    _config_lang = None  # 配置中的语言代码（如 zh-CN）
    _translations: Dict[str, Dict] = {}
    _fallback_locale = "zh_CN"
    _observers: List[Callable] = []
    _is_initialized = False

    # 配置语言代码 ↔ 翻译文件名映射
    CONFIG_TO_LOCALE = {
        "zh-CN": "zh_CN",
        "zh-TW": "zh_TW",
        "en-US": "en_US",
        "ja-JP": "ja_JP",
        "ko-KR": "ko_KR",
    }

    LOCALE_TO_CONFIG = {v: k for k, v in CONFIG_TO_LOCALE.items()}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._is_initialized:
            return
        self._is_initialized = True
        self._init_from_config()

    def _init_from_config(self):
        """从项目配置初始化"""
        config_lang = "en-US"  # 默认值

        try:
            # 检查配置是否已加载
            if g_config is not None:
                # 使用 getattr 避免直接访问未加载的配置
                config_lang = getattr(g_config, "app", {}).get("language", "en-US")
        except Exception:
            # 配置未加载或访问失败，使用默认值
            pass

        print(f"[I18n] Initializing with config language: {config_lang}")   
        self._config_lang = config_lang
        self._current_locale = self.CONFIG_TO_LOCALE.get(
            config_lang, self._fallback_locale
        )
        print(f"[I18n] Initializing with current_locale: {self._current_locale}")   


        self._load_all_translations()
        print(f"[I18n] Initialized with locale: {self._current_locale}")

    def _get_locales_dir(self) -> Path:
        """获取翻译文件目录"""
        return Path(__file__).parent / "locales"

    def _load_all_translations(self):
        """加载所有可用的翻译文件"""
        locales_dir = self._get_locales_dir()
        if not locales_dir.exists():
            print(f"[I18n] Locales directory not found: {locales_dir}")
            return

        for json_file in locales_dir.glob("*.json"):
            locale_code = json_file.stem
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    self._translations[locale_code] = json.load(f)
                    print(f"[I18n] Loaded: {locale_code}")
            except Exception as e:
                print(f"[I18n] Failed to load {json_file}: {e}")

        # 确保至少有回退语言
        if self._fallback_locale not in self._translations:
            self._translations[self._fallback_locale] = {}

    def set_language(self, config_lang_code: str, save_to_config: bool = True) -> bool:
        """
        切换语言

        Args:
            config_lang_code: 配置格式的语言代码 (zh-CN, en-US 等)
            save_to_config: 是否保存到配置文件

        Returns:
            bool: 是否切换成功
        """
        locale_code = self.CONFIG_TO_LOCALE.get(config_lang_code)
        if not locale_code:
            print(f"[I18n] Unknown language code: {config_lang_code}")
            return False

        if locale_code not in self._translations:
            print(f"[I18n] Translation not found: {locale_code}")
            return False

        old_locale = self._current_locale
        self._current_locale = locale_code
        self._config_lang = config_lang_code

        # 保存到项目配置
        if save_to_config and g_config:
            try:
                g_config.set("app.language", config_lang_code, save=True)
                print(f"[I18n] Saved language to config: {config_lang_code}")
            except Exception as e:
                print(f"[I18n] Failed to save to config: {e}")

        # 通知观察者
        if old_locale != locale_code:
            self._notify_observers(config_lang_code)

        return True

    def get_current_locale(self) -> str:
        """获取当前 locale 代码"""
        return self._current_locale

    def get_current_config_lang(self) -> str:
        """获取配置格式的语言代码"""
        return self._config_lang or self.LOCALE_TO_CONFIG.get(
            self._current_locale, "zh-CN"
        )

    def get_available_languages(self) -> List[Dict[str, str]]:
        """获取所有可用语言列表"""
        languages = []
        for locale_code in self._translations.keys():
            config_code = self.LOCALE_TO_CONFIG.get(locale_code, locale_code)
            # 尝试从翻译文件本身获取语言名称
            name = self._get_nested_value(
                self._translations[locale_code], "meta.language_name", config_code
            )
            languages.append(
                {"locale": locale_code, "config_code": config_code, "name": name}
            )
        return languages

    def add_observer(self, callback: Callable[[str], None]):
        """添加语言切换观察者"""
        if callback not in self._observers:
            self._observers.append(callback)

    def remove_observer(self, callback: Callable[[str], None]):
        """移除观察者"""
        if callback in self._observers:
            self._observers.remove(callback)

    def _notify_observers(self, new_lang: str):
        """通知所有观察者"""
        for callback in self._observers[:]:  # 复制列表避免修改时出错
            try:
                callback(new_lang)
            except Exception as e:
                print(f"[I18n] Observer error: {e}")

    def _get_nested_value(self, data: Dict, key_path: str, default: Any = None) -> Any:
        """获取嵌套字典值"""
        keys = key_path.split(".")
        value = data
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def get(self, key: str, default: Optional[str] = None, **kwargs) -> str:
        """
        获取翻译文本

        Args:
            key: 键路径，如 "app.title" 或 "sidebar.new_chat"
            default: 默认值
            **kwargs: 格式化参数

        Returns:
            str: 翻译后的文本
        """
        # 从当前语言获取
        value = self._get_nested_value(
            self._translations.get(self._current_locale, {}), key
        )

        # 回退到默认语言
        if value is None and self._current_locale != self._fallback_locale:
            value = self._get_nested_value(
                self._translations.get(self._fallback_locale, {}), key
            )

        # 使用默认值
        if value is None:
            value = default if default is not None else key

        # 格式化
        if kwargs and isinstance(value, str):
            try:
                # 支持 {name} 和 %(name)s 两种格式
                if "%(" in value:
                    return value % kwargs
                return value.format(**kwargs)
            except (KeyError, ValueError) as e:
                print(f"[I18n] Format error for key '{key}': {e}")

        return str(value)

    def get_with_plural(self, key: str, count: int, **kwargs) -> str:
        """
        获取带复数形式的翻译（用于消息数量等）

        翻译文件中需要定义：
        {
          "msg_count": {
            "one": "{count} 条消息",
            "other": "{count} 条消息"
          }
        }
        """
        plural_key = "one" if count == 1 else "other"
        full_key = f"{key}.{plural_key}"

        result = self.get(full_key, **kwargs)
        if result == full_key:  # 未找到，尝试获取默认形式
            result = self.get(key, **kwargs)

        return result.format(count=count, **kwargs)


# ============ 便捷函数 ============


def t(key: str, default: Optional[str] = None, **kwargs) -> str:
    """
    翻译函数简写

    用法:
        t("app.title")
        t("sidebar.msg_count", count=5)
        t("welcome.message", name="User", default="Hello!")
    """
    return I18n().get(key, default, **kwargs)


def set_language(config_lang_code: str, save_to_config: bool = True) -> bool:
    """切换语言"""
    return I18n().set_language(config_lang_code, save_to_config)


def get_current_language() -> str:
    """获取当前语言代码（配置格式）"""
    return I18n().get_current_config_lang()


def get_current_locale() -> str:
    """获取当前 locale 代码"""
    return I18n().get_current_locale()


def get_available_languages() -> List[Dict[str, str]]:
    """获取可用语言列表"""
    return I18n().get_available_languages()


def add_observer(callback: Callable[[str], None]):
    """添加语言切换观察者"""
    I18n().add_observer(callback)


def remove_observer(callback: Callable[[str], None]):
    """移除观察者"""
    I18n().remove_observer(callback)


def tp(key: str, count: int, **kwargs) -> str:
    """
    带复数的翻译

    用法:
        tp("sidebar.msg_count", 5)  # -> "5 条消息"
    """
    return I18n().get_with_plural(key, count, **kwargs)
