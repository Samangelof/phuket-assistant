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


# -------------------------------------------------------------------
class DataPlugin:
    """
    Абстракция для плагинов данных.
    """
    def is_relevant(self, query: str) -> bool:
        """Проверяет, связан ли запрос с данным типом данных"""
        raise NotImplementedError

    def get_data(self) -> str:
        """Получает данные для запроса"""
        raise NotImplementedError


class WeatherPlugin(DataPlugin):
    """Плагин для получения погоды"""
    def __init__(self):
        self.cache = {
            'timestamp': 0,
            'value': None
        }
        self.cache_ttl = 1800


    def is_relevant(self, query: str) -> bool:
        weather_keywords = {'погода', 'температур', 'дожд', 'солнц', '°c', 'осадк', 'ветер', 'влажност'}
        return any(keyword in query.lower() for keyword in weather_keywords)

    def extract_city(self, query: str) -> str:
        for word in query.split():
            if word.lower() in ['пхукет', 'бангкок', 'паттайя']:
                return word.capitalize()
        return "Пхукет" 

    def get_data(self, query: str) -> str:
        city = self.extract_city(query)
        if not city:
            return "[ОШИБКА ПОГОДЫ] Не удалось определить город. Нужно уточнить: 'Укажите город, например: `Погода в Пхукете`'"

        cache_key = f"weather_{city}"
        current_time = time.time()
        
        if cache_key in self.cache and current_time - self.cache[cache_key]['timestamp'] < self.cache_ttl:
            return self.cache[cache_key]['value']
        
        try:
            api_key = os.getenv("WEATHER_API_KEY")
            if not api_key:
                return "[ОШИБКА ПОГОДЫ] Сервис временно недоступен (отсутствует API-ключ)"
                
            url = f"https://api.weatherapi.com/v1/current.json?key={api_key}&q={city}&lang=ru"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            if "current" not in data:
                return f"[ОШИБКА ПОГОДЫ] Не удалось обработать данные для города '{city}'"
            
            current = data["current"]
            result = (
                f"Актуальная информация о погоде в {city} на сегодня: {current['temp_c']}°C, "
                f"{current['condition']['text']}. "
                f"Ветер: {current['wind_kph']/3.6:.1f} м/с"
            )
            
            self.cache[cache_key] = {
                'timestamp': current_time,
                'value': result
            }
            
            return result
            
        except requests.exceptions.Timeout:
            return "[ОШИБКА ПОГОДЫ] Сервис не отвечает. Попробуйте позже"
        except requests.exceptions.RequestException as e:
            return f"[ОШИБКА ПОГОДЫ] Ошибка соединения: {str(e)}"
        except Exception as e:
            return f"[ОШИБКА ПОГОДЫ] Неизвестная ошибка: {str(e)}"

class CurrencyPlugin(DataPlugin):
    """Плагин для получения информации о валюте"""
    def is_relevant(self, query: str) -> bool:
        currency_keywords = ["курс", "валюта", "доллар", "евро"]
        return any(word in query.lower() for word in currency_keywords)

    def get_data(self) -> str:
        # Ваш код для получения информации о валюте
        return "Курс доллара: 75 руб."


class NewsPlugin(DataPlugin):
    """Плагин для получения новостей"""
    def is_relevant(self, query: str) -> bool:
        news_keywords = ["новости", "события", "новость", "мировые новости"]
        return any(word in query.lower() for word in news_keywords)

    def get_data(self) -> str:
        # Ваш код для получения новостей
        return "Последние новости: важное событие в мире"
