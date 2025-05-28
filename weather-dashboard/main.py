import io
import os
import json
import logging
from datetime import datetime

import pandas as pd
import requests
import matplotlib
matplotlib.use('Agg')
from matplotlib.ticker import MultipleLocator

from flask import Flask, Response, render_template
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import calplot                     # for the cal-heatmap
import functions_framework 

# Configure logging
logging.basicConfig(level=logging.INFO)

# Your existing helper functions (convert_wind_direction, get_weather_data_from_cloud_function, etc.)
GET_WEATHER_DATA_CF_URL = "https://europe-west2-weathercloud-460719.cloudfunctions.net/get-weather-data"

def plot_data(dates, values, title, y_label):
    fig = Figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(dates, values)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.autofmt_xdate()
    return fig

def plot_bar(x, heights, title, y_label):
    fig = Figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(x, heights)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.autofmt_xdate()
    return fig

def plot_daily_bar(hours, values, title, y_label):
    fig = Figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(hours, values)
    ax.set_title(title)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel(y_label)
    return fig

def plot_24h_bar_greyed(xs, ys, title, ylabel):
    fig  = Figure(figsize=(13, 2.5))
    ax   = fig.add_subplot(1, 1, 1)
    nowh = datetime.now().hour

    # fill in missing hours with zero
    full_xs = list(range(24))
    full_ys = [0]*24
    for x, y in zip(xs, ys):
        full_ys[x] = y

    # rotate so “nowh − 1” is at the right edge
    ordered_hours = sorted(full_xs, key=lambda h: (h - 1 - nowh) % 24)
    ordered_ys    = [full_ys[h] for h in ordered_hours]

    # draw bars
    ax.bar(range(24), ordered_ys, color='black', width=0.5)
    ax.set_xlim(-0.5, 23.5)

    # label every 6 hours (using rotated positions)
    ticks  = [23, 17, 11, 5, 0]
    labels = [f"{ordered_hours[t]:02d}:00" for t in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)

    # shade “yesterday” up to midnight
    midnight_pos = ordered_hours.index(0)
    ax.axvspan(-0.5, midnight_pos, facecolor='lightgrey', alpha=0.5)

    # minor ticks and grid
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # y‐limits and label
    ax.set_ylim(0, max(1, ax.get_ylim()[1]))
    ax.set_ylabel(ylabel)

    # annotate “Yesterday” / “Today”
    ymax = ax.get_ylim()[1]
    ax.text(0,  ymax*0.95, 'Yesterday', ha='left',  va='top')
    ax.text(23, ymax*0.95, 'Today',     ha='right', va='top')

    # title & styling
    ax.set_title(title)
    ax.set_facecolor('#ffffff')
    fig.set_facecolor('#ffffff')
    fig.subplots_adjust(left=0.045, right=0.98, top=0.9, bottom=0.2)

    return fig

def plot_annual(df, agg_method, cmap_name):
    # df indexed by datetime, with a 'temperature' column
    yearly = getattr(df['temperature'].resample('M'), agg_method)()
    fig = Figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    yearly.plot(kind='bar', ax=ax, colormap=cmap_name)
    ax.set_title(f'Annual {agg_method.capitalize()} Temperatures by Month')
    ax.set_ylabel('Temperature (°C)')
    return fig

def convert_wind_direction(degrees):
    if degrees is None:
        return "N/A"
    directions = [
        "N", "NNE", "NNE", "NE", "NE", "ENE", "ENE", "E", "E", "ESE", "ESE", "SE", "SE", "SSE", "SSE", "S",
        "S", "SSW", "SSW", "SW", "SW", "WSW", "WSW", "W", "W", "WNW", "WNW", "NW", "NW", "NNW", "NNW", "N"
    ]
    index = int((degrees + 5.625) / 11.25) % 32
    return directions[index]

def get_weather_data_from_cloud_function(data_range: str) -> pd.DataFrame:
    """Fetch weather data from the get-weather-data Cloud Function."""
    logging.info(f"Attempting to fetch data for range: '{data_range}' from Cloud Function.")
    headers = {"Content-Type": "application/json"}
    payload = {"range": data_range}

    try:
        response = requests.post(GET_WEATHER_DATA_CF_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        json_data = response.json()

        if not json_data:
            logging.warning(f"Cloud Function returned no data for range: '{data_range}'.")
            return pd.DataFrame()

        df = pd.DataFrame(json_data)

        if 'timestamp_UTC' in df.columns:
            # parse as UTC, convert to Europe/London, then remove tz-info if you want naïve datetimes
            df['datetime'] = (
                pd.to_datetime(df['timestamp_UTC'], errors='coerce', utc=True)
                .dt.tz_convert('Europe/London')
                .dt.tz_localize(None)
            )
            df.dropna(subset=['datetime'], inplace=True)
            df.drop(columns=['timestamp_UTC'], inplace=True)
        else:
            logging.error("Cloud Function response missing 'timestamp_UTC' column.")
            return pd.DataFrame()

        if 'datetime' in df.columns:
            df = df.sort_values(by='datetime').reset_index(drop=True)
        else:
            logging.error("Processed DataFrame is missing 'datetime' column.")
            return pd.DataFrame()

        logging.info(f"Successfully fetched {len(df)} records for range: '{data_range}'.")
        return df

    except Exception as e:
        logging.error(f"Error fetching data for range '{data_range}': {e}")
        return pd.DataFrame()

def get_safe_summary_stat(df, field, op, default_value="N/A"):
    """Safely get summary statistics from DataFrame."""
    if df.empty or field not in df.columns:
        return default_value
    try:
        if op == 'max': return round(df[field].max(), 1)
        if op == 'min': return round(df[field].min(), 1)
        if op == 'sum': return round(df[field].sum(), 1)
        if op == 'mean': return round(df[field].mean(), 1)
        if op == 'max_rate': return round(df[field].max()*3600, 1)
        if op == 'max_speed_mph': return round(df[field].max()*2.23694, 1)
        return default_value
    except Exception as e:
        logging.error(f"Error calculating {op} for {field}: {e}")
        return default_value

# Create Flask app
app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    """Home route that displays weather dashboard."""
    # Fetch all data ranges from the Cloud Function
    latest_data = get_weather_data_from_cloud_function("latest")
    todays_data = get_weather_data_from_cloud_function("today")
    yesterdays_data = get_weather_data_from_cloud_function("yesterday")
    week_data = get_weather_data_from_cloud_function("week")
    month_data = get_weather_data_from_cloud_function("month")
    year_data = get_weather_data_from_cloud_function("year")

    # Prepare template variables with defaults
    template_vars = {
        "time_of_latest_reading": "N/A",
        "latest_temperature": "N/A",
        "latest_humidity": "N/A",
        "latest_rain_rate": "N/A",
        "latest_pressure": "N/A",
        "latest_luminance": "N/A",
        "latest_wind_speed_mph": "N/A",
        "latest_wind_direction": "N/A",
        "latest_wind_direction_converted": "N/A",
        "table": "<p>No annual data available.</p>",
        "num_rainy_days": "N/A",
        "total_days": "N/A",
        "rainy_percent": "N/A",
    }

    # Populate latest data
    if not latest_data.empty and 'datetime' in latest_data.columns:
        try:
            latest_row = latest_data.iloc[0]
            template_vars.update({
                "time_of_latest_reading": latest_row['datetime'].strftime('%A at %H:%M'),
                "latest_temperature": round(latest_row.get('temperature', float('nan')), 1),
                "latest_humidity": round(latest_row.get('humidity', float('nan')), 1),
                "latest_rain_rate": round(latest_row.get('rain_rate', float('nan')) * 3600, 1),
                "latest_pressure": round(latest_row.get('pressure', float('nan')), 1),
                "latest_luminance": round(latest_row.get('luminance', float('nan')), 1),
                "latest_wind_speed_mph": round(latest_row.get('wind_speed', float('nan')) * 2.23694, 1),
                "latest_wind_direction": latest_row.get('wind_direction', "N/A"),
                "latest_wind_direction_converted": convert_wind_direction(latest_row.get('wind_direction')),
            })
            
            # Handle NaN values
            for key in ["latest_temperature", "latest_humidity", "latest_rain_rate", "latest_pressure", "latest_luminance", "latest_wind_speed_mph"]:
                if pd.isna(template_vars[key]):
                    template_vars[key] = "N/A"

        except Exception as e:
            logging.error(f"Error processing latest data: {e}")

    # Populate summary statistics
    template_vars.update({
        "todays_max_temperature": get_safe_summary_stat(todays_data, 'temperature', 'max'),
        "todays_min_temperature": get_safe_summary_stat(todays_data, 'temperature', 'min'),
        "todays_total_rain": get_safe_summary_stat(todays_data, 'rain', 'sum'),
        "todays_average_pressure": get_safe_summary_stat(todays_data, 'pressure', 'mean'),
        "yesterdays_max_temperature": get_safe_summary_stat(yesterdays_data, 'temperature', 'max'),
        "yesterdays_min_temperature": get_safe_summary_stat(yesterdays_data, 'temperature', 'min'),
        "yesterdays_total_rain": get_safe_summary_stat(yesterdays_data, 'rain', 'sum'),
        "yesterdays_max_rain_rate": get_safe_summary_stat(yesterdays_data, 'rain_rate', 'max_rate'),
        "yesterdays_max_wind_speed": get_safe_summary_stat(yesterdays_data, 'wind_speed', 'max_speed_mph'),
        "weekly_max_temperature": get_safe_summary_stat(week_data, 'temperature', 'max'),
        "weekly_min_temperature": get_safe_summary_stat(week_data, 'temperature', 'min'),
        "weekly_total_rain": get_safe_summary_stat(week_data, 'rain', 'sum'),
        "weekly_max_rain_rate": get_safe_summary_stat(week_data, 'rain_rate', 'max_rate'),
        "weekly_max_wind_speed": get_safe_summary_stat(week_data, 'wind_speed', 'max_speed_mph'),
        "monthly_max_temperature": get_safe_summary_stat(month_data, 'temperature', 'max'),
        "monthly_min_temperature": get_safe_summary_stat(month_data, 'temperature', 'min'),
        "monthly_total_rain": get_safe_summary_stat(month_data, 'rain', 'sum'),
        "monthly_max_rain_rate": get_safe_summary_stat(month_data, 'rain_rate', 'max_rate'),
        "monthly_max_wind_speed": get_safe_summary_stat(month_data, 'wind_speed', 'max_speed_mph'),
        "annual_max_temperature": get_safe_summary_stat(year_data, 'temperature', 'max'),
        "annual_min_temperature": get_safe_summary_stat(year_data, 'temperature', 'min'),
        "annual_total_rain": get_safe_summary_stat(year_data, 'rain', 'sum'),
        "annual_max_rain_rate": get_safe_summary_stat(year_data, 'rain_rate', 'max_rate'),
        "annual_max_wind_speed": get_safe_summary_stat(year_data, 'wind_speed', 'max_speed_mph'),
    })

    return render_template("current.html", **template_vars)

@app.route('/plot_temperature.png')
def plot_temperature_png():
    data = get_weather_data_from_cloud_function("last7days")
    if data.empty or 'datetime' not in data.columns or 'temperature' not in data.columns:
        return Response("No data for temperature plot.", mimetype='text/plain', status=204)
    fig = plot_data(data['datetime'], data['temperature'].rolling(window=5, min_periods=1).mean(), 'Temperature', 'Temperature (C)')
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/plot_humidity.png')
def plot_humidity_png():
    data = get_weather_data_from_cloud_function("last7days")
    if data.empty or 'datetime' not in data.columns or 'humidity' not in data.columns:
        return Response("No data for humidity plot.", mimetype='text/plain', status=204)
    fig = plot_data(data['datetime'], data['humidity'].rolling(window=5, min_periods=1).mean(), 'Humidity', 'Humidity (%)')
    fig.gca().set_ylim(0,100)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/plot_pressure.png')
def plot_pressure_png():
    data = get_weather_data_from_cloud_function("last7days")
    if data.empty or 'datetime' not in data.columns or 'pressure' not in data.columns:
        return Response("No data for pressure plot.", mimetype='text/plain', status=204)
    fig = plot_data(data['datetime'], data['pressure'], 'Pressure', 'Pressure (hPa)')
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/plot_daily_rainfall.png')
def plot_daily_rainfall_png():
    todays_data = get_weather_data_from_cloud_function("today")
    if todays_data.empty or 'datetime' not in todays_data.columns or 'rain' not in todays_data.columns:
        return Response("No data for daily rainfall plot.", mimetype='text/plain', status=204)
    
    # Ensure 'datetime' is datetime type for dt accessor
    if not pd.api.types.is_datetime64_any_dtype(todays_data['datetime']):
        todays_data['datetime'] = pd.to_datetime(todays_data['datetime'], errors='coerce')
        todays_data.dropna(subset=['datetime'], inplace=True)
        if todays_data.empty:
             return Response("No valid datetime data for daily rainfall plot.", mimetype='text/plain', status=204)

    rain_data = todays_data.groupby(todays_data['datetime'].dt.hour)['rain'].sum()
    fig = plot_daily_bar(rain_data.index, rain_data.values, "Today's Hourly Rainfall", "Rainfall (mm)")
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/plot_24h_rainfall.png')
def plot_24h_rainfall_png():
    last24h_data = get_weather_data_from_cloud_function("last24h")
    if last24h_data.empty or 'datetime' not in last24h_data.columns or 'rain' not in last24h_data.columns:
        return Response("No data for last 24h rainfall plot.", mimetype='text/plain', status=204)

    # ensure datetime dtype
    if not pd.api.types.is_datetime64_any_dtype(last24h_data['datetime']):
        last24h_data['datetime'] = pd.to_datetime(last24h_data['datetime'], errors='coerce')
        last24h_data.dropna(subset=['datetime'], inplace=True)
        if last24h_data.empty:
            return Response("No valid datetime data for 24h rainfall plot.", mimetype='text/plain', status=204)

    # sum rain by hour
    rain_data = last24h_data.groupby(last24h_data['datetime'].dt.hour)['rain'].sum()

    fig = plot_24h_bar_greyed(
        list(rain_data.index),
        list(rain_data.values),
        "Hourly Rainfall",
        "Rainfall (mm)"
    )
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/plot_rain.png')
def plot_rain_png():
    last_7_days_data = get_weather_data_from_cloud_function("last7days")
    if last_7_days_data.empty or 'datetime' not in last_7_days_data.columns or 'rain' not in last_7_days_data.columns:
        return Response("No data for weekly rain plot.", mimetype='text/plain', status=204)

    if not pd.api.types.is_datetime64_any_dtype(last_7_days_data['datetime']):
        last_7_days_data['datetime'] = pd.to_datetime(last_7_days_data['datetime'], errors='coerce')
        last_7_days_data.dropna(subset=['datetime'], inplace=True)
        if last_7_days_data.empty:
            return Response("No valid datetime data for weekly rain plot.", mimetype='text/plain', status=204)
            
    rain_data = last_7_days_data.groupby(last_7_days_data['datetime'].dt.date)['rain'].sum()
    # rain_data.index is currently date objects, convert to datetime for plotting consistency if needed by plot_bar
    rain_data.index = pd.to_datetime(rain_data.index) # Convert date objects to datetime64[ns]
    
    fig = plot_bar(rain_data.index, rain_data.values, "Last 7 Days Rainfall", "Rainfall (mm)")
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/plot_annual_max_temperatures.png')
def plot_annual_max_temperatures_png():
    data = get_weather_data_from_cloud_function("year")
    if data.empty or 'datetime' not in data.columns:
        return Response("No data for annual max temp plot.", mimetype='text/plain', status=204)
    if not pd.api.types.is_datetime64_any_dtype(data['datetime']):
        data['datetime'] = pd.to_datetime(data['datetime'], errors='coerce')
        data.dropna(subset=['datetime'], inplace=True)
        if data.empty:
            return Response("No valid datetime data for annual max temp plot.", mimetype='text/plain', status=204)
            
    data.set_index('datetime', inplace = True)
    fig = plot_annual(data, 'max', 'coolwarm') # plot_annual checks for 'temperature' column
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/plot_annual_rain_days.png')
def plot_annual_rain_days_png():
    data = get_weather_data_from_cloud_function("year")
    if data.empty or 'datetime' not in data.columns or 'rain' not in data.columns:
        return Response("No data for annual rain days plot.", mimetype='text/plain', status=204)

    if not pd.api.types.is_datetime64_any_dtype(data['datetime']):
        data['datetime'] = pd.to_datetime(data['datetime'], errors='coerce')
        data.dropna(subset=['datetime'], inplace=True)
        if data.empty:
             return Response("No valid datetime data for annual rain days plot.", mimetype='text/plain', status=204)

    data.set_index('datetime', inplace=True)

    # Resample daily, sum rain, check if > 1mm, convert boolean to int (0 or 1)
    daily_rain_events = data['rain'].resample('D').sum() > 1.0 
    rain_event_data = daily_rain_events.astype(int) # Series of 0s and 1s

    cmap = mcolors.ListedColormap(['#f0f0f0', 'black']) # 0 maps to lightgrey, 1 maps to black

    fig_to_return = None
    if rain_event_data.empty:
        logging.info("No daily rain event data after resampling for calplot.")
        fig = Figure(figsize=(13,2.5))
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'No data for annual rain days plot after resampling', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_axis_off()
        fig.set_facecolor('#ffffff')
        fig_to_return = fig
    else:
        logging.info(f"Plotting calplot for rain_event_data with {len(rain_event_data)} days.")
        # calplot expects a Series. how='sum' is okay here even for 0/1 data.
        # vmin/vmax ensures the colormap is used correctly for 0 and 1.
        fig, axis_list = calplot.calplot(data=rain_event_data, how='sum', figsize=(13, 2.5), 
                                       cmap=cmap, linecolor='#ffffff', yearlabels=True, 
                                       colorbar=False, vmin=0, vmax=1)
        fig.set_facecolor('#ffffff')
        if isinstance(axis_list, plt.Axes): # Single axis
            axis_list.set_facecolor('#ffffff')
        elif isinstance(axis_list, (list, pd.Series, pd.Index)): # List or array of axes
            for ax_row in axis_list:
                if hasattr(ax_row, '__iter__'): 
                     for ax in ax_row:
                        if isinstance(ax, plt.Axes):
                            ax.set_facecolor('#ffffff')
                elif isinstance(ax_row, plt.Axes): 
                     ax_row.set_facecolor('#ffffff')
        fig_to_return = fig
    
    plt.rcParams['font.family'] = 'DejaVu Sans'

    output = io.BytesIO()
    FigureCanvas(fig_to_return).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/plot_annual_min_temperatures.png')
def plot_annual_min_temperatures_png():
    data = get_weather_data_from_cloud_function("year")
    if data.empty or 'datetime' not in data.columns:
         return Response("No data for annual min temp plot.", mimetype='text/plain', status=204)

    if not pd.api.types.is_datetime64_any_dtype(data['datetime']):
        data['datetime'] = pd.to_datetime(data['datetime'], errors='coerce')
        data.dropna(subset=['datetime'], inplace=True)
        if data.empty:
            return Response("No valid datetime data for annual min temp plot.", mimetype='text/plain', status=204)

    data.set_index('datetime', inplace = True)
    fig = plot_annual(data, 'min', 'Blues') # plot_annual checks for 'temperature' column
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

# Cloud Functions entry point
@functions_framework.http
def display_weather_data(request):
    """Cloud Function entry point."""
    # Create a real Flask request context from the incoming WSGI environ
    with app.request_context(request.environ):
        try:
            return app.full_dispatch_request()
        except Exception as e:
            logging.error(f"Error in Cloud Function: {e}")
            return f"Error: {str(e)}", 500