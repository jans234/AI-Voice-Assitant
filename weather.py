
import requests
from geopy.geocoders import Nominatim

def get_location(location_name):
    geolocator = Nominatim(user_agent="geoapi")
    location = geolocator.geocode(location_name)
    return location.latitude, location.longitude


def celsius_to_fahrenheit(celsius):
    return (celsius * 9/5) + 32
    
def get_weather(location_name):
    lat, lon = get_location(location_name)
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,apparent_temperature,is_day,precipitation,rain,showers,snowfall,weather_code,cloud_cover,pressure_msl,surface_pressure,wind_speed_10m,wind_direction_10m,wind_gusts_10m&daily=temperature_2m_max,temperature_2m_min,daylight_duration,uv_index_max"

    response = requests.get(url)

    if response.status_code == 200:
        response = response.json()
        pretty_response = f"""Current Weather in {location_name}:
            - Temperature:  {response['current']['temperature_2m']}°C / {celsius_to_fahrenheit(response['current']['temperature_2m']):.1f}°F
            - Relative Humidity: {response['current']['relative_humidity_2m']}%
            - Apparent Temperature: {response['current']['apparent_temperature']}°C
            - Precipitation: {response['current']['precipitation']} mm
            - Cloud Cover: {response['current']['cloud_cover']}%
            - Wind Speed: {response['current']['wind_speed_10m']} km/h
            - Wind Direction: {response['current']['wind_direction_10m']}°

            Forecast for the next 7 days:
            """

        for i, day in enumerate(response['daily']['time']):
                max_temp = response['daily']['temperature_2m_max'][i]
                min_temp = response['daily']['temperature_2m_min'][i]
                uv_index = response['daily']['uv_index_max'][i]
                daylight = response['daily']['daylight_duration'][i]
                
                pretty_response += f"""
                {day}:
                - Max Temp: {max_temp}°C
                - Min Temp: {min_temp}°C
                - UV Index: {uv_index}
                - Daylight Duration: {daylight / 3600:.2f} hours
                """
        return pretty_response        
