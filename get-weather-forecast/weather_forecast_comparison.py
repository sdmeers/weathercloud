import requests
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

GET_DATA_URL = "https://europe-west2-weathercloud-460719.cloudfunctions.net/get-weather-data"
GET_FORECAST_URL = "https://europe-west2-weathercloud-460719.cloudfunctions.net/get-weather-forecast"

payload = {
    "range": "last24h",
    "fields": ["temperature", "pressure", "humidity"]
}

# Fetch actual weather data
try:
     print(f"Fetching actual data from {GET_DATA_URL}...")
     response_actual = requests.post(GET_DATA_URL, json=payload)
     response_actual.raise_for_status()  # Raise an exception for bad status codes
     actual_data = response_actual.json()
     print(f"Successfully fetched {len(actual_data)} actual weather readings.")
except requests.exceptions.RequestException as e:
     print(f"Error fetching actual data: {e}")
     actual_data = []

# Fetch forecast data
try:
    print(f"\nFetching forecast data from {GET_FORECAST_URL}...")
    response_forecast = requests.post(GET_FORECAST_URL, json=payload)
    response_forecast.raise_for_status()
    forecast_data = response_forecast.json()
    print(f"Successfully fetched {len(forecast_data)} forecast hours.")
except requests.exceptions.RequestException as e:
    print(f"Error fetching forecast data: {e}")
    forecast_data = []

# Data Processing
if actual_data and forecast_data:
    # Create DataFrames from the JSON responses
    df_actual = pd.DataFrame(actual_data)
    df_forecast = pd.DataFrame(forecast_data)
     
    # Convert the 'time' columns to proper datetime objects
    df_actual['timestamp_UTC'] = pd.to_datetime(df_actual['timestamp_UTC'])
    df_forecast['time'] = pd.to_datetime(df_forecast['time'])
     
    # Set the 'time' column as the index for both DataFrames
    df_actual.set_index('timestamp_UTC', inplace=True)
    df_forecast.set_index('time', inplace=True)
   
    # Rename columns for clarity before combining
    df_actual.rename(columns={'temperature': 'Actual_temperature', 'pressure': 'Actual_pressure', 'humidity': 'Actual_humidity'}, inplace=True)
    df_forecast.rename(columns={'temperature': 'Forecast_temperature', 'pressure': 'Forecast_pressure', 'humidity': 'Forecast_humidity'}, inplace=True)
    
    # Combine the two data series into a single DataFrame for easy plotting
    df_comparison = pd.concat([df_actual, df_forecast], axis=1)
    
    print("\nData prepared for plotting")
else:
    print("\nCould not process data due to fetch errors.")

if 'df_comparison' in locals() and not df_comparison.empty:
    # Forward-fill forecast data to ensure lines are continuous
    for col in ['Forecast_temperature', 'Forecast_pressure', 'Forecast_humidity']:
        df_comparison[col] = df_comparison[col].ffill()

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(3, 1, figsize=(15, 21), sharex=True)
    
    # Plot Temperature
    axes[0].plot(df_comparison.index, df_comparison['Actual_temperature'], label='Actual Temperature', marker='.', linestyle='-', color='royalblue')
    axes[0].plot(df_comparison.index, df_comparison['Forecast_temperature'], label='Forecast Temperature', marker='x', linestyle='--', color='darkorange')
    axes[0].set_title('Actual vs. Forecast Temperature', fontsize=14)
    axes[0].set_ylabel('Temperature (°C)', fontsize=12)
    axes[0].legend()
    axes[0].grid(True)

    # Plot Pressure
    axes[1].plot(df_comparison.index, df_comparison['Actual_pressure'], label='Actual Pressure', marker='.', linestyle='-', color='green')
    axes[1].plot(df_comparison.index, df_comparison['Forecast_pressure'], label='Forecast Pressure', marker='x', linestyle='--', color='red')
    axes[1].set_title('Actual vs. Forecast Pressure', fontsize=14)
    axes[1].set_ylabel('Pressure (hPa)', fontsize=12)
    axes[1].legend()
    axes[1].grid(True)

    # Plot Humidity
    axes[2].plot(df_comparison.index, df_comparison['Actual_humidity'], label='Actual Humidity', marker='.', linestyle='-', color='purple')
    axes[2].plot(df_comparison.index, df_comparison['Forecast_humidity'], label='Forecast Humidity', marker='x', linestyle='--', color='brown')
    axes[2].set_title('Actual vs. Forecast Humidity', fontsize=14)
    axes[2].set_ylabel('Humidity (%)', fontsize=12)
    axes[2].legend()
    axes[2].grid(True)

    # Common formatting
    fig.suptitle('Actual vs. Forecast Weather Data (Last 24 Hours)', fontsize=18, y=0.95)
    axes[2].set_xlabel('Time', fontsize=12)
    
    # Format the x-axis to show hours and minutes nicely
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[2].xaxis.set_major_locator(mdates.HourLocator(interval=2))
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('weather_comparison.png')
    print("Plot saved to weather_comparison.png")
else:
    print("No data to plot.")