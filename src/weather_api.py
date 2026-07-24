import requests


def get_current_weather(latitude, longitude):

    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={latitude}"
        f"&longitude={longitude}"
        f"&current=temperature_2m,relative_humidity_2m,"
        f"wind_speed_10m,rain"
        f"&timezone=auto"
    )

    try:

        response = requests.get(url, timeout=30)

        response.raise_for_status()

        data = response.json()

        current = data["current"]

        return {

            "temperature": current.get("temperature_2m", 0),

            "humidity": current.get(
                "relative_humidity_2m",
                0
            ),

            "rainfall": current.get(
                "rain",
                0
            ),

            "wind_speed": current.get(
                "wind_speed_10m",
                0
            )

        }

    except Exception as e:

        print("Weather API Error:", e)

        return {

            "temperature": 0,

            "humidity": 0,

            "rainfall": 0,

            "wind_speed": 0

        }