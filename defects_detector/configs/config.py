from typing import Any


class Configuration:
    """Класс для хранения конфигурации детектора"""

    def __init__(self, **kwargs):
        """Инициализирует конфигурацию из словаря параметров"""
        pass

    def get(self, key: str, default: Any = None) -> Any:
        """Получает значение параметра по ключу"""
        pass

    def save(self, file_name: str) -> None:
        """Сохраняет конфигурацию в файл"""
        pass

    @staticmethod
    def load(file_name: str) -> 'Configuration':
        """Загружает конфигурацию из файла"""
        pass