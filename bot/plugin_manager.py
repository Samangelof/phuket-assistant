import json
import requests
import os
import time
import re
from plugins.gtts_text_to_speech import GTTSTextToSpeech
from plugins.auto_tts import AutoTextToSpeech
from plugins.dice import DicePlugin
from plugins.youtube_audio_extractor import YouTubeAudioExtractorPlugin
from plugins.ddg_image_search import DDGImageSearchPlugin
from plugins.spotify import SpotifyPlugin
from plugins.crypto import CryptoPlugin
from plugins.weather import WeatherPlugin
from plugins.ddg_web_search import DDGWebSearchPlugin
from plugins.wolfram_alpha import WolframAlphaPlugin
from plugins.deepl import DeeplTranslatePlugin
from plugins.worldtimeapi import WorldTimeApiPlugin
from plugins.whois_ import WhoisPlugin
from plugins.webshot import WebshotPlugin
from plugins.iplocation import IpLocationPlugin


class PluginManager:
    """
    A class to manage the plugins and call the correct functions
    """

    def __init__(self, config):
        enabled_plugins = config.get('plugins', [])
        plugin_mapping = {
            'wolfram': WolframAlphaPlugin,
            'weather': WeatherPlugin,
            'crypto': CryptoPlugin,
            'ddg_web_search': DDGWebSearchPlugin,
            'ddg_image_search': DDGImageSearchPlugin,
            'spotify': SpotifyPlugin,
            'worldtimeapi': WorldTimeApiPlugin,
            'youtube_audio_extractor': YouTubeAudioExtractorPlugin,
            'dice': DicePlugin,
            'deepl_translate': DeeplTranslatePlugin,
            'gtts_text_to_speech': GTTSTextToSpeech,
            'auto_tts': AutoTextToSpeech,
            'whois': WhoisPlugin,
            'webshot': WebshotPlugin,
            'iplocation': IpLocationPlugin,
        }
        self.plugins = [plugin_mapping[plugin]() for plugin in enabled_plugins if plugin in plugin_mapping]

    def get_functions_specs(self):
        """
        Return the list of function specs that can be called by the model
        """
        return [spec for specs in map(lambda plugin: plugin.get_spec(), self.plugins) for spec in specs]

    async def call_function(self, function_name, helper, arguments):
        """
        Call a function based on the name and parameters provided
        """
        plugin = self.__get_plugin_by_function_name(function_name)
        if not plugin:
            return json.dumps({'error': f'Function {function_name} not found'})
        return json.dumps(await plugin.execute(function_name, helper, **json.loads(arguments)), default=str)

    def get_plugin_source_name(self, function_name) -> str:
        """
        Return the source name of the plugin
        """
        plugin = self.__get_plugin_by_function_name(function_name)
        if not plugin:
            return ''
        return plugin.get_source_name()

    def __get_plugin_by_function_name(self, function_name):
        return next((plugin for plugin in self.plugins
                    if function_name in map(lambda spec: spec.get('name'), plugin.get_spec())), None)



# # -------------------------------------------------------------------
# class DataPlugin:
#     """
#     Абстракция для плагинов данных.
#     """
#     def is_relevant(self, query: str) -> bool:
#         """Проверяет, связан ли запрос с данным типом данных"""
#         raise NotImplementedError

#     def get_data(self) -> str:
#         """Получает данные для запроса"""
#         raise NotImplementedError


# class WeatherPlugin(DataPlugin):
#     """Плагин для получения погоды с фокусом на Таиланд и пляжные курорты"""
    
#     def __init__(self):
#         # Инициализация кеша для каждого города отдельно
#         self.cache = {}
#         self.cache_ttl = 1800  # 30 минут
        
#         # Словарь локаций в Таиланде и их вариаций написания
#         self.thai_locations = {
#             # Основные курорты и города
#             'пхукет': 'Пхукет', 'пукет': 'Пхукет', 'пухкет': 'Пхукет', 'пхукете': 'Пхукет',
#             'phuket': 'Пхукет', 'пхукете': 'Пхукет', 'пхукету': 'Пхукет',
            
#             'паттай': 'Паттайя', 'патай': 'Паттайя', 'патайя': 'Паттайя', 'паттая': 'Паттайя', 
#             'патая': 'Паттайя', 'pattaya': 'Паттайя', 'паттайе': 'Паттайя', 'паттайю': 'Паттайя',
            
#             'бангкок': 'Бангкок', 'bangkok': 'Бангкок', 'бангкоке': 'Бангкок', 'бкк': 'Бангкок',
            
#             'самуи': 'Самуи', 'самуй': 'Самуи', 'самуе': 'Самуи', 'koh samui': 'Самуи',
#             'ко самуи': 'Самуи', 'острове самуи': 'Самуи', 'самуйи': 'Самуи',
            
#             'краби': 'Краби', 'krabi': 'Краби', 'краби провинции': 'Краби', 'ао нанге': 'Краби',
            
#             'хуахин': 'Хуахин', 'hua hin': 'Хуахин', 'хуа хин': 'Хуахин', 'хуахине': 'Хуахин',
            
#             'чиангмай': 'Чиангмай', 'чианг май': 'Чиангмай', 'чианг-май': 'Чиангмай', 
#             'chiang mai': 'Чиангмай', 'чиангмае': 'Чиангмай',
            
#             'паи': 'Паи', 'pai': 'Паи',
#             'пхи-пхи': 'Пхи-Пхи', 'пхипхи': 'Пхи-Пхи', 'phi phi': 'Пхи-Пхи', 'ко пхи пхи': 'Пхи-Пхи',
#             'ко чанг': 'Ко Чанг', 'чанг': 'Ко Чанг', 'koh chang': 'Ко Чанг',
#             'ко ланта': 'Ко Ланта', 'ланта': 'Ко Ланта', 'koh lanta': 'Ко Ланта',
            
#             # Общие фразы о Таиланде
#             'тайланд': 'Бангкок', 'таиланд': 'Бангкок', 'thailand': 'Бангкок', 
#             'тай': 'Бангкок', 'тае': 'Бангкок'
#         }
        
#         # Ключевые слова для погоды, включая русские и тайские 
#         self.weather_keywords = {
#             'погод', 'температур', 'дожд', 'солнц', '°c', 'осадк', 'ветер', 'влажност',
#             'жара', 'жарко', 'градус', 'облач', 'облак', 'тепло', 'холодно',
#             'климат', 'сезон', 'метео', 'гроза', 'ливень', 'прогноз', 'weather',
#             'метеоданн', 'зонт', 'осадки', 'туман', 'солнце'
#         }

#     def is_relevant(self, query: str) -> bool:
#         """
#         Определяет, относится ли запрос к погоде в Таиланде
#         """
#         query_lower = query.lower()
        
#         # Проверяем наличие ключевых слов погоды
#         has_weather_keyword = any(keyword in query_lower for keyword in self.weather_keywords)
        
#         # Проверяем явные запросы о погоде
#         explicit_weather_questions = [
#             'какая погода', 'что с погодой', 'погода в', 'погода на', 
#             'какой прогноз', 'какая температура', 'сколько градусов'
#         ]
#         explicit_question = any(phrase in query_lower for phrase in explicit_weather_questions)
        
#         # Проверяем наличие тайской локации
#         has_thai_location = self._extract_location(query_lower) is not None
        
#         # Считаем запрос релевантным, если:
#         # 1. Есть прямой вопрос о погоде (независимо от указания локации)
#         # 2. Есть упоминание погоды и локации в Таиланде
#         return explicit_question or (has_weather_keyword and has_thai_location)

#     def _extract_location(self, query: str) -> str:
#         """
#         Извлекает название местоположения из запроса с учетом предлогов и контекста
#         """
#         # Паттерны с предлогами для более точного определения
#         location_patterns = [
#             r'погода (?:в|на) (\w+[-]?\w*)',  # погода в Пхукете, погода на Пхи-Пхи
#             r'(?:в|на) (\w+[-]?\w*) (?:погода|градус|температур)',  # в Пхукете погода
#             r'(?:погода|температура) (?:в городе|в|на острове) (\w+[-]?\w*)',  # погода в городе Краби
#             r'(?:температура|градус|тепло|жарко|холодно) (?:в|на) (\w+[-]?\w*)'  # температура в Паттайе
#         ]
        
#         # Сначала проверяем по паттернам
#         for pattern in location_patterns:
#             matches = re.findall(pattern, query)
#             if matches:
#                 location_word = matches[0].lower()
#                 # Проверяем совпадение с нашим словарем локаций
#                 for key, value in self.thai_locations.items():
#                     if key in location_word:
#                         return value
        
#         # Если по паттернам не нашли, проверяем просто наличие слов из словаря
#         for key, value in self.thai_locations.items():
#             if key in query:
#                 return value
        
#         # Проверяем общие обозначения текущей локации
#         location_here_patterns = ['здесь', 'тут', 'у нас', 'у меня', 'сейчас']
#         if any(pattern in query for pattern in location_here_patterns):
#             # По умолчанию для чата ПХУКЕТ
#             return "Пхукет"
                
#         return None

#     def get_data(self, query: str) -> str:
#         """
#         Получает и форматирует данные о погоде для найденной локации
#         """
#         # Определяем местоположение
#         location = self._extract_location(query.lower())
#         if not location:
#             # Для чата ПХУКЕТ используем Пхукет по умолчанию
#             location = "Пхукет"
        
#         # Проверяем кеш
#         cache_key = f"weather_{location.lower()}"
#         current_time = time.time()
        
#         if cache_key in self.cache and current_time - self.cache[cache_key]['timestamp'] < self.cache_ttl:
#             return self.cache[cache_key]['value']
        
#         try:
#             api_key = os.getenv("WEATHER_API_KEY")
#             if not api_key:
#                 # Более информативное сообщение об ошибке
#                 return "[ПОГОДА] API недоступен. Данных нет."
            
#             # Используем официальное название для API запроса
#             location_api_map = {
#                 'Пхукет': 'Phuket',
#                 'Паттайя': 'Pattaya',
#                 'Бангкок': 'Bangkok',
#                 'Самуи': 'Koh Samui',
#                 'Краби': 'Krabi',
#                 'Хуахин': 'Hua Hin',
#                 'Чиангмай': 'Chiang Mai',
#                 'Паи': 'Pai',
#                 'Пхи-Пхи': 'Phi Phi Islands',
#                 'Ко Чанг': 'Koh Chang',
#                 'Ко Ланта': 'Koh Lanta'
#             }
            
#             api_location = location_api_map.get(location, location)
#             url = f"https://api.weatherapi.com/v1/current.json?key={api_key}&q={api_location}&lang=ru"
            
#             response = requests.get(url, timeout=5)
#             response.raise_for_status()
#             data = response.json()
            
#             if "current" not in data:
#                 return f"[ПОГОДА] Нет данных для {location}."
            
#             # Извлекаем данные
#             current = data["current"]
#             condition = current['condition']['text']
            
#             # Формируем базовую информацию
#             result = (
#                 f"[ПОГОДА] В {location} сейчас {current['temp_c']}°C. "
#                 f"{condition}. "
#                 f"Ветер: {current['wind_kph']/3.6:.1f} м/с. "
#             )
            
#             # Добавляем важную информацию для пляжного отдыха
#             if 'humidity' in current:
#                 result += f"Влажность: {current['humidity']}%. "
            
#             if 'feelslike_c' in current and abs(current['feelslike_c'] - current['temp_c']) > 1:
#                 result += f"Ощущается как {current['feelslike_c']}°C. "
            
#             if 'uv' in current:
#                 uv = current['uv']
#                 uv_description = "низкий"
#                 if uv > 2 and uv <= 5:
#                     uv_description = "умеренный"
#                 elif uv > 5 and uv <= 7:
#                     uv_description = "высокий"
#                 elif uv > 7 and uv <= 10:
#                     uv_description = "очень высокий"
#                 elif uv > 10:
#                     uv_description = "экстремальный"
                
#                 result += f"UV-индекс: {uv} ({uv_description}). "
            
#             # Если есть вероятность осадков, добавляем информацию
#             if 'precip_mm' in current and current['precip_mm'] > 0:
#                 result += f"Осадки: {current['precip_mm']} мм. "
            
#             # Кешируем результат
#             self.cache[cache_key] = {
#                 'timestamp': current_time,
#                 'value': result.strip()
#             }
            
#             return result.strip()
            
#         except requests.exceptions.Timeout:
#             return "[ПОГОДА] Сервис не отвечает. Данных нет."
#         except requests.exceptions.RequestException as e:
#             return f"[ПОГОДА] Ошибка получения данных: {str(e)}"
#         except Exception as e:
#             return f"[ПОГОДА] Ошибка: {str(e)}"

# class CurrencyPlugin(DataPlugin):
#     """Плагин для получения информации о валюте"""
#     def is_relevant(self, query: str) -> bool:
#         currency_keywords = ["курс", "валюта", "доллар", "евро"]
#         return any(word in query.lower() for word in currency_keywords)

#     def get_data(self) -> str:
#         # Ваш код для получения информации о валюте
#         return "Курс доллара: 75 руб."


# class NewsPlugin(DataPlugin):
#     """Плагин для получения новостей"""
#     def is_relevant(self, query: str) -> bool:
#         news_keywords = ["новости", "события", "новость", "мировые новости"]
#         return any(word in query.lower() for word in news_keywords)

#     def get_data(self) -> str:
#         # Ваш код для получения новостей
#         return "Последние новости: важное событие в мире"
