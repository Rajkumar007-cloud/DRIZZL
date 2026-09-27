import pandas as pd
import numpy as np

np.random.seed(42)

rows = 3000

data = []

for _ in range(rows):

    temperature = np.random.uniform(20, 38)
    humidity = np.random.uniform(50, 100)
    wind = np.random.uniform(2, 30)
    pressure = np.random.uniform(980, 1015)

    regime = np.random.choice(
        ["Active", "Break", "Depression"]
    )

    if regime == "Active":
        nwp_rain = np.random.uniform(30, 120)
        actual = nwp_rain + np.random.uniform(10, 25)

    elif regime == "Break":
        nwp_rain = np.random.uniform(0, 30)
        actual = nwp_rain + np.random.uniform(-5, 5)

    else:
        nwp_rain = np.random.uniform(50, 200)
        actual = nwp_rain + np.random.uniform(20, 50)

    data.append([
        temperature,
        humidity,
        wind,
        pressure,
        nwp_rain,
        actual,
        regime
    ])

df = pd.DataFrame(
    data,
    columns=[
        "temperature",
        "humidity",
        "wind",
        "pressure",
        "nwp_rainfall",
        "actual_rainfall",
        "regime"
    ]
)

df.to_csv("data/weather_data.csv", index=False)

print("Dataset generated!")