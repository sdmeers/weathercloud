import io
import os
import json
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import matplotlib
matplotlib.use('Agg')

from matplotlib.ticker import MultipleLocator
from matplotlib.dates import DayLocator, DateFormatter

from flask import Flask, Response, render_template
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

import calplot
import functions_framework

# Configure logging
logging.basicConfig(level=logging.INFO)

GET_WEATHER_DATA_CF_URL = "https://europe-west2-weathercloud-460719.cloudfunctions.net/get-weather-data"


def convert_wind_direction(deg):
    """
    Converts a wind direction in degrees (0-360, where 0=SE, increasing clockwise)
    to a cardinal or intercardinal direction (N, NE, E, SE, S, SW, W, NW).
    
    This function assumes a specific mapping based on the provided degrees:
    0 = SE, 45 = S, 90 = SW, 135 = W, 180 = NW, 225 = N, 270 = NE, 315 = E.
    
    Args:
        deg (int): Wind direction in degrees.
        
    Returns:
        str: Cardinal or intercardinal direction string.
        
    Raises:
        ValueError: If the degree value does not match one of the predefined values.
    """
    if deg == 225: return "N"
    if deg == 180: return "NW"
    if deg == 135: return "W"
    if deg == 90:  return "SW"
    if deg == 45:  return "S"
    if deg == 0:   return "SE"
    if deg == 315: return "E"
    if deg == 270: return "NE"
    raise ValueError("Error: Invalid wind direction")


def get_weather_data_from_cloud_function(data_range: str, fields: list = None) -> pd.DataFrame:
    """
    Fetches weather data from the Cloud Function.
    
    Args:
        data_range (str): The time range for the data (e.g., "latest", "today", "last7days", "year").
        fields (list, optional): A list of specific fields to request. If None, all fields
                                  will be requested (though the backend might still filter).
                                  'timestamp_UTC' is always added if not present.
    Returns:
        pd.DataFrame: A pandas DataFrame containing the fetched and processed weather data.
    """
    logging.info(f"Fetching '{data_range}' with fields: {fields}…")
    
    payload = {"range": data_range}
    if fields:
        # Ensure timestamp_UTC is always requested as it's needed for processing
        if 'timestamp_UTC' not in fields:
            fields.insert(0, 'timestamp_UTC')
        payload["fields"] = fields

    try:
        r = requests.post(GET_WEATHER_DATA_CF_URL,
                          headers={"Content-Type": "application/json"},
                          json=payload, timeout=30)
        r.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
        data = r.json()
        if not data:
            logging.info(f"No data returned for '{data_range}' with fields: {fields}")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        
        # Parse UTC → London local, drop tz
        # Ensure 'timestamp_UTC' exists before attempting to convert
        if 'timestamp_UTC' in df.columns:
            df['datetime'] = (
                pd.to_datetime(df['timestamp_UTC'], errors='coerce', utc=True)
                .dt.tz_convert('Europe/London')
                .dt.tz_localize(None)
            )
            df.dropna(subset=['datetime'], inplace=True)
            df.drop(columns=['timestamp_UTC'], inplace=True)
            df.sort_values('datetime', inplace=True)
            df.reset_index(drop=True, inplace=True)
        else:
            logging.warning(f"'timestamp_UTC' not found in data for range '{data_range}'. This might lead to issues.")
            # If timestamp_UTC is missing, we can't create 'datetime' column, return empty or handle differently
            return pd.DataFrame() 

        return df

    except requests.exceptions.Timeout:
        logging.error(f"Timeout fetching '{data_range}' from Cloud Function.")
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        logging.error(f"Request error fetching '{data_range}': {e}")
        return pd.DataFrame()
    except json.JSONDecodeError:
        logging.error(f"JSON decode error for '{data_range}'. Response: {r.text}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Unexpected error fetching '{data_range}': {e}", exc_info=True)
        return pd.DataFrame()


def get_safe_summary_stat(df, field, op, default_value="N/A"):
    if df.empty or field not in df.columns:
        return default_value
    try:
        val = {
            'max':  df[field].max(),
            'min':  df[field].min(),
            'sum':  df[field].sum(),
            'mean': df[field].mean(),
            'max_rate': df[field].max() * 3600,
            'max_speed_mph': df[field].max() * 2.23694
        }[op]
        return round(val, 1)
    except Exception as e:
        logging.error(f"Stat error {op} on {field}: {e}")
        return default_value


# ——————————————————————————————————————————————————————————————————————————————
# Plotting helpers
# ——————————————————————————————————————————————————————————————————————————————

def plot_data(xs, ys, title, ylabel):
    """
    Last 7 days line plots: black line, day‐of‐week ticks, vertical gridlines.
    """
    fig = Figure()
    ax  = fig.add_subplot(1, 1, 1)
    ax.plot(xs, ys, 'k')
    ax.xaxis.set_major_locator(DayLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%a'))
    ax.grid(axis='x', linestyle='--', alpha=0.7, which='major')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    return fig


def plot_bar(xs, ys, title, ylabel):
    """
    Last 7 days rainfall bar: black bars, day‐of‐week ticks.
    """
    fig = Figure()
    ax  = fig.add_subplot(1, 1, 1)
    ax.bar(xs, ys, color='black')
    ax.xaxis.set_major_locator(DayLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%a'))
    ax.set_ylabel(ylabel)
    # force y≥1
    y0, y1 = ax.get_ylim()
    ax.set_ylim(0, max(1, y1))
    ax.set_title(title)
    fig.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    return fig


def plot_daily_bar(hours, values, title, ylabel):
    fig = Figure(figsize=(13, 2.5))
    ax  = fig.add_subplot(1, 1, 1)
    ax.bar(hours, values, color='black', width=0.5)

    ticks  = list(range(0, 25, 6))
    labels = [f"{h:02d}:00" for h in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    ax.set_ylabel(ylabel)
    y0, y1 = ax.get_ylim()
    ax.set_ylim(0, max(1, y1))

    ax.set_title(title)
    fig.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    return fig


def plot_24h_bar_greyed(xs, ys, title, ylabel):
    fig = Figure(figsize=(13, 2.5))
    ax  = fig.add_subplot(1, 1, 1)

    nowh = datetime.now(ZoneInfo("Europe/London")).hour

    full_xs = list(range(24))
    full_ys = [0]*24
    for x, y in zip(xs, ys):
        full_ys[x] = y

    ordered_hours = sorted(full_xs, key=lambda h: (h - 1 - nowh) % 24)
    ordered_ys    = [full_ys[h] for h in ordered_hours]

    ax.bar(range(24), ordered_ys, color='black', width=0.5)
    ax.set_xlim(-0.5, 23.5)

    ticks  = [23, 17, 11, 5, 0]
    labels = [f"{ordered_hours[t]:02d}:00" for t in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)

    midnight_pos = ordered_hours.index(0)
    ax.axvspan(-0.5, midnight_pos, facecolor='lightgrey', alpha=0.5)

    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    ax.set_ylabel(ylabel)
    y0, y1 = ax.get_ylim()
    ax.set_ylim(0, max(1, y1))

    ymax = ax.get_ylim()[1]
    ax.text(0,   ymax*0.95, 'Yesterday', ha='left',   va='top')
    ax.text(23, ymax*0.95, 'Today',     ha='right', va='top')

    ax.set_title(title)
    fig.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    fig.subplots_adjust(left=0.045, right=0.98, top=0.9, bottom=0.2)

    return fig

def plot_annual(data, how, cmap):
    fig, axis = calplot.calplot(data = data['temperature'], how = how, figsize=(13,2.5) , cmap = cmap, linecolor='#ffffff', yearlabels=True, colorbar=False, textformat='{:.0f}')#,suptitle = title)
    fig.set_size_inches(13, 2.5)
    fig.set_facecolor('#ffffff')
    for ax in axis.flatten():
        ax.set_facecolor('#ffffff')
    plt.rcParams['font.family'] = 'DejaVu Sans'
    return fig

# ——————————————————————————————————————————————————————————————————————————————
# Routes
# ——————————————————————————————————————————————————————————————————————————————

app = Flask(__name__)


@app.route("/")
@app.route("/home")
def home():
    # Define fields needed for each section of the home page
    # These lists are for the get_weather_data_from_cloud_function call
    
    # Fields for 'latest' reading (current conditions)
    latest_fields = [
        'timestamp_UTC', 'temperature', 'humidity', 'pressure', 'rain_rate',
        'luminance', 'wind_speed', 'wind_direction'
    ]
    latest = get_weather_data_from_cloud_function("latest", fields=latest_fields)

    # Fields for daily summaries (today, yesterday)
    daily_summary_fields = [
        'timestamp_UTC', 'temperature', 'rain', 'pressure', 'rain_rate', 'wind_speed'
    ]
    today = get_weather_data_from_cloud_function("today", fields=daily_summary_fields)
    yesterday = get_weather_data_from_cloud_function("yesterday", fields=daily_summary_fields)

    # Fields for weekly/monthly summaries
    period_summary_fields = [
        'timestamp_UTC', 'temperature', 'rain', 'rain_rate', 'wind_speed'
    ]
    week = get_weather_data_from_cloud_function("week", fields=period_summary_fields)
    month = get_weather_data_from_cloud_function("month", fields=period_summary_fields)

    # Fields for annual summaries (rainy days calculation, annual stats)
    annual_summary_fields = [
        'timestamp_UTC', 'temperature', 'rain', 'rain_rate', 'wind_speed'
    ]
    year = get_weather_data_from_cloud_function("year", fields=annual_summary_fields)

    vars = {
        "page_title": "Weather Summary",
        "page_icon": "fa-solid fa-dashboard",
        "time_of_latest_reading": "N/A",
        "latest_temperature":     "N/A",
        "latest_humidity":        "N/A",
        "latest_rain_rate":       "N/A",
        "latest_pressure":        "N/A",
        "latest_luminance":       "N/A",
        "latest_wind_speed_mph":  "N/A",
        "latest_wind_direction":          "N/A",
        "latest_wind_direction_converted":"N/A",
        "table": "<p>No annual data available.</p>", # This seems to be a placeholder, not used by calplot
        "num_rainy_days": "N/A",
        "total_days":     "N/A",
        "rainy_percent":  "N/A",
    }

    if not latest.empty:
        row = latest.iloc[0]
        vars.update({
            "time_of_latest_reading": row['datetime'].strftime('%A at %H:%M'),
            "latest_temperature":     round(row['temperature'], 1),
            "latest_humidity":        round(row['humidity'],    1),
            "latest_rain_rate":       round(row['rain_rate']*3600, 1),
            "latest_pressure":        round(row['pressure'],    1),
            "latest_luminance":       round(row['luminance'],   1),
            "latest_wind_speed_mph":  round(row['wind_speed']*2.23694, 1),
            "latest_wind_direction":          row.get('wind_direction', "N/A"),
            "latest_wind_direction_converted":convert_wind_direction(row.get('wind_direction')),
        })

    # summary stats…
    vars.update({
        "todays_max_temperature":   get_safe_summary_stat(today,     'temperature', 'max'),
        "todays_min_temperature":   get_safe_summary_stat(today,     'temperature', 'min'),
        "todays_total_rain":        get_safe_summary_stat(today,     'rain',        'sum'),
        "todays_average_pressure":  get_safe_summary_stat(today,     'pressure',    'mean'),
        "yesterdays_max_temperature": get_safe_summary_stat(yesterday,'temperature','max'),
        "yesterdays_min_temperature": get_safe_summary_stat(yesterday,'temperature','min'),
        "yesterdays_total_rain":    get_safe_summary_stat(yesterday,'rain',        'sum'),
        "yesterdays_max_rain_rate":   get_safe_summary_stat(yesterday,'rain_rate',  'max_rate'),
        "yesterdays_max_wind_speed":  get_safe_summary_stat(yesterday,'wind_speed', 'max_speed_mph'),
        "weekly_max_temperature":     get_safe_summary_stat(week,      'temperature','max'),
        "weekly_min_temperature":     get_safe_summary_stat(week,      'temperature','min'),
        "weekly_total_rain":          get_safe_summary_stat(week,      'rain',       'sum'),
        "weekly_max_rain_rate":       get_safe_summary_stat(week,      'rain_rate',  'max_rate'),
        "weekly_max_wind_speed":      get_safe_summary_stat(week,      'wind_speed', 'max_speed_mph'),
        "monthly_max_temperature":    get_safe_summary_stat(month,     'temperature','max'),
        "monthly_min_temperature":    get_safe_summary_stat(month,     'temperature','min'),
        "monthly_total_rain":         get_safe_summary_stat(month,     'rain',       'sum'),
        "monthly_max_rain_rate":      get_safe_summary_stat(month,     'rain_rate',  'max_rate'),
        "monthly_max_wind_speed":     get_safe_summary_stat(month,     'wind_speed', 'max_speed_mph'),
        "annual_max_temperature":     get_safe_summary_stat(year,      'temperature','max'),
        "annual_min_temperature":     get_safe_summary_stat(year,      'temperature','min'),
        "annual_total_rain":          get_safe_summary_stat(year,      'rain',       'sum'),
        "annual_max_rain_rate":       get_safe_summary_stat(year,      'rain_rate',  'max_rate'),
        "annual_max_wind_speed":      get_safe_summary_stat(year,      'wind_speed', 'max_speed_mph'),
    })

    if not year.empty:
        # how many days in the year_data had >1mm rain?
        # Ensure 'rain' column exists before accessing
        if 'rain' in year.columns:
            daily = year.set_index('datetime')['rain'].resample('D').sum()
            num_rainy = int((daily > 1.0).sum())
            total     = len(daily)  # number of days we have
            pct       = round(100 * num_rainy / total) if total else 0

            vars.update({
                "num_rainy_days": num_rainy,
                "total_days":     total,
                "rainy_percent":  pct,
            })
        else:
            logging.warning("No 'rain' column in annual data for rainy day calculation.")

    return render_template("current.html", **vars)


@app.route('/plot_temperature.png')
def plot_temperature_png():
    d = get_weather_data_from_cloud_function("last7days", fields=['temperature'])
    if d.empty: return Response(status=204)
    fig = plot_data(d['datetime'], d['temperature'].rolling(5, min_periods=1).mean(),
                    'Temperature', 'Temperature (°C)')
    buf = io.BytesIO(); FigureCanvas(fig).print_png(buf)
    return Response(buf.getvalue(), mimetype='image/png')


@app.route('/plot_humidity.png')
def plot_humidity_png():
    d = get_weather_data_from_cloud_function("last7days", fields=['humidity'])
    if d.empty: return Response(status=204)
    fig = plot_data(d['datetime'], d['humidity'].rolling(5, min_periods=1).mean(),
                    'Humidity', 'Relative Humidity (%)')
    fig.axes[0].set_ylim(0, 100)
    buf = io.BytesIO(); FigureCanvas(fig).print_png(buf)
    return Response(buf.getvalue(), mimetype='image/png')


@app.route('/plot_pressure.png')
def plot_pressure_png():
    d = get_weather_data_from_cloud_function("last7days", fields=['pressure'])
    if d.empty: return Response(status=204)
    fig = plot_data(d['datetime'], d['pressure'],
                    'Pressure', 'Mean Sea Level Pressure (hPa)')
    buf = io.BytesIO(); FigureCanvas(fig).print_png(buf)
    return Response(buf.getvalue(), mimetype='image/png')


@app.route('/plot_rain.png')
def plot_rain_png():
    d = get_weather_data_from_cloud_function("last7days", fields=['rain'])
    if d.empty: return Response(status=204)
    rain = d.groupby(d['datetime'].dt.date)['rain'].sum()
    rain.index = pd.to_datetime(rain.index)
    fig = plot_bar(rain.index, rain.values,
                   'Last 7 Days Rainfall', 'Rainfall (mm)')
    buf = io.BytesIO(); FigureCanvas(fig).print_png(buf)
    return Response(buf.getvalue(), mimetype='image/png')


@app.route('/plot_daily_rainfall.png')
def plot_daily_rainfall_png():
    d = get_weather_data_from_cloud_function("today", fields=['rain'])
    if d.empty: return Response(status=204)
    d = d.set_index('datetime')
    hourly = d['rain'].resample('H').sum()
    fig = plot_daily_bar(hourly.index.hour, hourly.values,
                         "Today's Hourly Rainfall", "Rainfall (mm)")
    buf = io.BytesIO(); FigureCanvas(fig).print_png(buf)
    return Response(buf.getvalue(), mimetype='image/png')


@app.route('/plot_24h_rainfall.png')
def plot_24h_rainfall_png():
    d = get_weather_data_from_cloud_function("last24h", fields=['rain'])
    if d.empty: return Response(status=204)
    rain = d.groupby(d['datetime'].dt.hour)['rain'].sum()
    fig = plot_24h_bar_greyed(rain.index.tolist(), rain.values.tolist(),
                              "Hourly Rainfall", "Rainfall (mm)")
    buf = io.BytesIO(); FigureCanvas(fig).print_png(buf)
    return Response(buf.getvalue(), mimetype='image/png')


@app.route('/plot_annual_max_temperatures.png')
def plot_annual_max_temperatures_png():
    d = get_weather_data_from_cloud_function("year", fields=['temperature'])
    if d.empty: return Response(status=204)
    d.set_index('datetime', inplace=True)
    fig = plot_annual(d, 'max', 'coolwarm')
    buf = io.BytesIO(); FigureCanvas(fig).print_png(buf)
    return Response(buf.getvalue(), mimetype='image/png')


@app.route('/plot_annual_min_temperatures.png')
def plot_annual_min_temperatures_png():
    d = get_weather_data_from_cloud_function("year", fields=['temperature'])
    if d.empty: return Response(status=204)
    d.set_index('datetime', inplace=True)
    fig = plot_annual(d, 'min', 'Blues')
    buf = io.BytesIO(); FigureCanvas(fig).print_png(buf)
    return Response(buf.getvalue(), mimetype='image/png')


@app.route('/plot_annual_rain_days.png')
def plot_annual_rain_days_png():
    d = get_weather_data_from_cloud_function("year", fields=['rain'])
    if d.empty: return Response(status=204)
    d.set_index('datetime', inplace=True)
    # Ensure 'rain' column exists before accessing
    if 'rain' in d.columns:
        rain_events = (d['rain'].resample('D').sum() > 1).astype(int)
        cmap = mcolors.ListedColormap(['#f0f0f0','black'])
        fig, axes = calplot.calplot(
            data=rain_events, how='sum',
            figsize=(13, 2.5),
            cmap=cmap, linecolor='#ffffff',
            yearlabels=True, colorbar=False,
            vmin=0, vmax=1
        )
        fig.set_size_inches(13, 2.5)
        fig.set_facecolor('#ffffff')
        for ax in (axes if hasattr(axes, 'flatten') else [axes]):
            ax.set_facecolor('#ffffff')
        plt.rcParams['font.family'] = 'DejaVu Sans'

        buf = io.BytesIO(); FigureCanvas(fig).print_png(buf)
        return Response(buf.getvalue(), mimetype='image/png')
    else:
        logging.warning("No 'rain' column available for annual rain days plot.")
        return Response(status=204) # No content if data is missing


@functions_framework.http
def display_weather_data(request):
    with app.request_context(request.environ):
        try:
            return app.full_dispatch_request()
        except Exception as e:
            logging.error(f"Cloud Function error: {e}", exc_info=True)
            return f"Error: {e}", 500