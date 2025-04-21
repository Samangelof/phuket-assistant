import os
import requests
import re


def get_data(city: str = "Phuket"):
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



def extract_city(query: str) -> str:
    pattern = r"(в|на)\s([а-яА-Я\- ]+)"
    match = re.search(pattern, query.lower())
    if match:
        return match.group(2).strip()
    return "пхукет"

print(extract_city('погода в тайланде в пхукете'))