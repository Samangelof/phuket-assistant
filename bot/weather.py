import time
import requests
import os


class WeatherCache:
    def __init__(self, ttl=1200):  # ttl = время жизни кеша в секундах (600 секунд = 20 минут)
        self.cache = {}
        self.ttl = ttl  # Время жизни данных в кешe

    def get_weather(self):
        current_time = time.time()
        if "weather_data" in self.cache:
            cached_time, data = self.cache["weather_data"]
            # Проверяем, не устарели ли данные
            if current_time - cached_time < self.ttl:
                return data  # Возвращаем кешированные данные, если они ещё актуальны
        return None

    def set_weather(self, data):
        self.cache["weather_data"] = (time.time(), data)

    def is_weather_related(self, query: str) -> bool:
        """Связан ли запрос с погодой."""
        weather_keywords = ["погода", "температура", "дождь", "снег", "солнечно", "влажность"]
        return any(word in query.lower() for word in weather_keywords)
    


async def get_weather_from_api(city: str = "Phuket"):
    api_key = os.getenv("WEATHER_API_KEY")  
    url = f"https://api.weatherapi.com/v1/current.json?key={api_key}&q={city}&lang=ru"
    response = requests.get(url)
    data = response.json()
    if "current" in data:
        temp = data["current"]["temp_c"]
        condition = data["current"]["condition"]["text"]
        return f"Сейчас в {city}: {temp}°C, {condition.lower()}."
    else:
        return "Не удалось получить данные о погоде."