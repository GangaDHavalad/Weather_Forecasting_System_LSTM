import pandas as pd
import requests
from datetime import datetime, timedelta
import os
import time

os.makedirs("data/raw", exist_ok=True)

districts = pd.read_csv("dataset/districts.csv")

end_date = datetime.today()
start_date = end_date - timedelta(days=5 * 365)

all_data = []

TOTAL = len(districts)

for index, row in districts.iterrows():

    district = row["district"]
    state = row["state"]
    lat = row["latitude"]
    lon = row["longitude"]

    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}"
        f"&longitude={lon}"
        f"&start_date={start_date.strftime('%Y-%m-%d')}"
        f"&end_date={end_date.strftime('%Y-%m-%d')}"
        "&daily=temperature_2m_mean,"
        "relative_humidity_2m_mean,"
        "precipitation_sum,"
        "wind_speed_10m_max"
        "&timezone=auto"
    )

    response = None

    # Retry up to 5 times
    for attempt in range(5):

        try:

            response = requests.get(url, timeout=60)

            if response.status_code == 429:

                wait_time = 30

                print(
                    f"Rate limit reached for {district}. Waiting {wait_time} seconds..."
                )

                time.sleep(wait_time)

                continue

            response.raise_for_status()

            break

        except requests.exceptions.RequestException as e:

            print(f"Retry {attempt+1}/5 -> {district}: {e}")

            time.sleep(10)

    if response is None or response.status_code != 200:

        print(f"Skipped {district}")

        continue

    try:

        data = response.json()

        if "daily" not in data:
            print(f"No daily data for {district}")
            continue

        df = pd.DataFrame({
            "date": data["daily"]["time"],
            "temperature": data["daily"]["temperature_2m_mean"],
            "humidity": data["daily"]["relative_humidity_2m_mean"],
            "rainfall": data["daily"]["precipitation_sum"],
            "wind_speed": data["daily"]["wind_speed_10m_max"],
        })

        df["district"] = district
        df["state"] = state

        all_data.append(df)

        print(f"[{index+1}/{TOTAL}] Downloaded {district}")

    except Exception as e:

        print(f"Error processing {district}: {e}")

    # Wait before next request
    time.sleep(3)

if len(all_data) == 0:
    print("No data downloaded.")
    exit()

weather = pd.concat(all_data, ignore_index=True)

weather.to_csv(
    "data/raw/weather_dataset.csv",
    index=False
)

print("\nDataset Saved Successfully!")
print(weather.head())