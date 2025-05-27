# main.py
import datetime
import io
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Use 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator, AutoLocator
from matplotlib.dates import DayLocator

import calplot # Assuming this is installed
from flask import Flask, render_template, Response, url_for
import requests # NEW: for making HTTP requests to the Cloud Function
import json # For handling JSON decoding errors

import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logging.getLogger('requests').setLevel(logging.WARNING) # Suppress verbose requests logging

app = Flask(__name__)

# --- Configuration for your get-weather-data Cloud Function ---
# IMPORTANT: Replace this with the actual URL of your deployed get-weather-data Cloud Function
# Consider making this an environment variable for better practice
GET_WEATHER_DATA_CF_URL = "https://europe-west2-weathercloud-460719.cloudfunctions.net/get-weather-data"

# --- Helper Function: convert_wind_direction (from weather_helper.py) ---
def convert_wind_direction(degrees):
    if degrees is None:
        return "N/A"
    # Using 32 directions as implied by the original code's list length,
    # though only 16 unique strings are provided.
    directions = [
        "N", "NNE", "NNE", "NE", "NE", "ENE", "ENE", "E", "E", "ESE", "ESE", "SE", "SE", "SSE", "SSE", "S",
        "S", "SSW", "SSW", "SW", "SW", "WSW", "WSW", "W", "W", "WNW", "WNW", "NW", "NW", "NNW", "NNW", "N"
    ]
    # For 16 standard compass points, use: index = int((degrees + 11.25) / 22.5) % 16
    # For 32 finer points, use: index = int((degrees + 5.625) / 11.25) % 32
    # Sticking to the previous interpretation of 32 for index calculation, even if string mapping is simpler.
    index = int((degrees + 5.625) / 11.25) % 32
    return directions[index]

# --- NEW Helper Function to fetch data from your get-weather-data Cloud Function ---
def get_weather_data_from_cloud_function(data_range: str) -> pd.DataFrame:
    """
    Fetches weather data from the get-weather-data Cloud Function.

    Args:
        data_range (str): The range argument to send to the Cloud Function
                          (e.g., "latest", "today", "last24h", "yesterday",
                           "last7days", "week", "month", "year").

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the fetched weather data.
                      'timestamp_UTC' from the CF response is converted to datetime
                      and the original 'timestamp_UTC' column is dropped.
                      The new datetime column is named 'datetime'.
                      Returns an empty DataFrame on error or no data.
    """
    logging.info(f"Attempting to fetch data for range: '{data_range}' from Cloud Function.")
    headers = {"Content-Type": "application/json"}
    payload = {"range": data_range}

    try:
        response = requests.post(GET_WEATHER_DATA_CF_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        json_data = response.json()

        if not json_data:
            logging.warning(f"Cloud Function returned no data for range: '{data_range}'.")
            return pd.DataFrame() # Return empty DataFrame if no data

        df = pd.DataFrame(json_data)

        if 'timestamp_UTC' in df.columns:
            # Convert 'timestamp_UTC' to datetime objects, placing them in a new 'datetime' column
            df['datetime'] = pd.to_datetime(df['timestamp_UTC'], errors='coerce')
            
            # Drop rows where the conversion to datetime failed (NaT values)
            df.dropna(subset=['datetime'], inplace=True)
            
            # Drop the original 'timestamp_UTC' string column as it has been processed
            df.drop(columns=['timestamp_UTC'], inplace=True)
        else:
            logging.error("Cloud Function response missing 'timestamp_UTC' column. Check CF output.")
            return pd.DataFrame()

        # Ensure 'datetime' column exists after processing before trying to sort
        if 'datetime' in df.columns:
            df = df.sort_values(by='datetime').reset_index(drop=True) # Sort by datetime
        else:
            # This case should ideally not be reached if the above logic is correct
            # and timestamp_UTC was present, but as a safeguard:
            logging.error("Processed DataFrame is missing 'datetime' column before sorting.")
            return pd.DataFrame()


        logging.info(f"Successfully fetched {len(df)} records for range: '{data_range}'.")
        return df

    except requests.exceptions.RequestException as e:
        logging.error(f"Network or HTTP error fetching data from Cloud Function for range '{data_range}': {e}")
        return pd.DataFrame()
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON response from Cloud Function for range '{data_range}': {e}")
        # You might want to inspect response.text here if it's not valid JSON
        # logging.debug(f"Non-JSON response for range '{data_range}': {response.text}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"An unexpected error occurred in get_weather_data_from_cloud_function for range '{data_range}': {e}", exc_info=True)
        return pd.DataFrame()

# --- Helper for fetching and getting summary stats gracefully ---
def get_safe_summary_stat(df, field, op, default_value="N/A"):
    if df.empty or field not in df.columns:
        return default_value
    try:
        # Ensure the column is numeric before applying operations
        if not pd.api.types.is_numeric_dtype(df[field]):
            # Attempt conversion, or log and return default if not appropriate
            # For now, assuming it should be numeric or operation will fail gracefully
            pass

        if op == 'max': return round(df[field].max(), 1)
        if op == 'min': return round(df[field].min(), 1)
        if op == 'sum': return round(df[field].sum(), 1)
        if op == 'mean': return round(df[field].mean(), 1)
        if op == 'max_rate': return round(df[field].max()*3600, 1) # Assuming conversion
        if op == 'max_speed_mph': return round(df[field].max()*2.23694, 1) # Assuming conversion
        return default_value
    except Exception as e:
        logging.error(f"Error calculating {op} for {field}: {e}")
        return default_value

# --- Plotting Functions (Updated to use Cloud Function data) ---
def plot_data(xs, ys, title, ylabel):
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.plot(xs, ys, 'k')
    axis.xaxis.set_major_formatter(mdates.DateFormatter('%a'))
    axis.xaxis.set_major_locator(DayLocator())
    axis.grid(axis='x', linestyle='--', alpha=0.7, which='major')
    axis.set_ylabel(ylabel)
    fig.set_facecolor('#ffffff')
    axis.set_facecolor('#ffffff')
    axis.set_title(title)
    return fig

def plot_bar(xs, ys, title, ylabel):
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.bar(xs, ys, color=['black'])
    axis.xaxis.set_major_formatter(mdates.DateFormatter('%a'))
    axis.xaxis.set_major_locator(mdates.DayLocator())
    axis.set_ylabel(ylabel)
    y_min, y_max = axis.get_ylim()
    axis.set_ylim(0, max(1, y_max)) # Ensure y_max is at least 1 if data is all zero
    fig.set_facecolor('#ffffff')
    axis.set_facecolor('#ffffff')
    axis.set_title(title)
    return fig

def plot_daily_bar(xs, ys, title, ylabel):
    fig = Figure(figsize=(13, 2.5))
    axis = fig.add_subplot(1, 1, 1)
    axis.bar(xs, ys, color='black', width=0.5)
    ticks = range(0, 25, 6)
    labels = [f"{tick:02d}:00" for tick in ticks]
    axis.set_xticks(ticks)
    axis.set_xticklabels(labels)
    axis.xaxis.set_minor_locator(MultipleLocator(1))
    axis.grid(axis='y', linestyle='--', alpha=0.7)
    axis.set_ylabel(ylabel)
    y_min, y_max = axis.get_ylim()
    axis.set_ylim(0, max(1, y_max)) # Ensure y_max is at least 1
    fig.set_facecolor('#ffffff')
    axis.set_facecolor('#ffffff')
    axis.set_title(title)
    return fig

def plot_24h_bar_greyed(xs, ys, title, ylabel):
    fig = Figure(figsize=(13, 2.5))
    axis = fig.add_subplot(1, 1, 1)
    current_hour = datetime.datetime.now().hour # Consider timezone implications if deployed

    full_xs_indices = list(range(24)) # These are indices 0-23 for plotting
    full_ys = [0]*24
    
    # Map actual hour data (xs) to the full_ys array
    if xs is not None and ys is not None:
        for x_hour, y_val in zip(xs, ys):
            if isinstance(x_hour, (int, float)) and 0 <= x_hour < 24:
                full_ys[int(x_hour)] = y_val

    # Order hours for plotting: current hour last on the right
    # The lambda sorts hours so that (current_hour - 1) % 24 is plotted last.
    # Example: if current_hour = 10, 9 will be last.
    # Plotting sequence starts from current_hour, wraps around.
    ordered_hours_values = sorted(full_xs_indices, key=lambda h_idx: (h_idx - current_hour) % 24)
    
    ordered_ys_values = [full_ys[h_idx] for h_idx in ordered_hours_values]

    bar_width = 0.5
    # Plot bars using 0..23 as x-coordinates for the bars
    axis.bar(full_xs_indices, ordered_ys_values, color='black', width=bar_width)
    axis.set_xlim(-0.5, 23.5)
    
    # Set tick labels based on the actual hour values in ordered_hours_values
    # Show ticks for specific bar positions (e.g., 0th bar, 6th bar, etc.)
    tick_positions_on_plot = [0, 6, 12, 18, 23] # Indices of bars on the plot
    labels = []
    for plot_idx in tick_positions_on_plot:
        if 0 <= plot_idx < len(ordered_hours_values):
            actual_hour = ordered_hours_values[plot_idx]
            labels.append(f"{actual_hour:02d}:00")
        else:
            labels.append("") 
            
    axis.set_xticks(tick_positions_on_plot)
    axis.set_xticklabels(labels)
    
    # Find where midnight (hour 0) is in the ordered plot
    midnight_actual_hour = 0
    midnight_bar_plot_position = -1
    if midnight_actual_hour in ordered_hours_values:
        midnight_bar_plot_position = ordered_hours_values.index(midnight_actual_hour)

    if midnight_bar_plot_position != -1:
        # Shade from start of plot up to the right edge of the bar representing midnight
        axis.axvspan(-0.5, midnight_bar_plot_position + bar_width/2, facecolor='lightgrey', alpha=0.5)
    
    axis.xaxis.set_minor_locator(MultipleLocator(1))
    axis.grid(axis='y', linestyle='--', alpha=0.7)
    axis.set_ylabel(ylabel)
    axis.set_ylim(0, max(1, axis.get_ylim()[1]))
    
    # Adjust text positions if needed
    axis.text(0.5, axis.get_ylim()[1] * 0.95, 'Yesterday', ha='left', va='top', transform=axis.transAxes) # Use transAxes for relative positioning
    axis.text(0.95, axis.get_ylim()[1] * 0.95, 'Today', ha='right', va='top', transform=axis.transAxes) # Use transAxes

    fig.subplots_adjust(left=0.05, right=0.98, top=0.9, bottom=0.2) # Adjusted left for y-label
    fig.set_facecolor('#ffffff')
    axis.set_facecolor('#ffffff')
    axis.set_title(title)
    return fig


def plot_annual(data, how, cmap):
    # Ensure the index is a datetime index for calplot
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    
    # calplot requires non-empty data in the specific column
    if 'temperature' not in data.columns or data['temperature'].empty:
        fig = Figure(figsize=(13,2.5))
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'No temperature data for annual plot', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_axis_off()
        fig.set_facecolor('#ffffff') # Set facecolor for the figure as well
        return fig

    fig, axis_list = calplot.calplot(data = data['temperature'], how = how,  figsize=(13,2.5) , cmap = cmap, linecolor='#ffffff', yearlabels=True, colorbar=False, textformat='{:.0f}')
    fig.set_facecolor('#ffffff')
    # calplot returns a list of axes, iterate to set facecolor
    if isinstance(axis_list, plt.Axes): # Single axis
        axis_list.set_facecolor('#ffffff')
    elif isinstance(axis_list, (list, pd.Series, pd.Index)): # List or array of axes
        for ax_row in axis_list: # calplot can return array of arrays for multiple years
            if hasattr(ax_row, '__iter__'): # If it's a row of axes
                 for ax in ax_row:
                    if isinstance(ax, plt.Axes):
                        ax.set_facecolor('#ffffff')
            elif isinstance(ax_row, plt.Axes): # If it's a single axis in the list
                 ax_row.set_facecolor('#ffffff')

    plt.rcParams['font.family'] = 'DejaVu Sans' # Ensure font is available or fallback
    return fig


# --- Flask Routes ---

@app.route("/")
@app.route("/home")
def home():
    # Fetch all data ranges from the Cloud Function
    # Consider error handling or alternative display if any of these fail
    latest_data = get_weather_data_from_cloud_function("latest")
    todays_data = get_weather_data_from_cloud_function("today")
    yesterdays_data = get_weather_data_from_cloud_function("yesterday")
    week_data = get_weather_data_from_cloud_function("week")
    month_data = get_weather_data_from_cloud_function("month")
    year_data = get_weather_data_from_cloud_function("year")

    # Prepare data for rendering, with N/A for missing values
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
        # Default annual table
        "table": "<p>No annual data available.</p>",
        "num_rainy_days": "N/A",
        "total_days": "N/A",
        "rainy_percent": "N/A",
        "dashboard_URL": url_for('home'), # Dynamic URL for dashboard
        "index_URL": url_for('home') # Pointing to home for simplicity, or could be a /data route
    }

    # Populate latest data
    if not latest_data.empty and 'datetime' in latest_data.columns:
        try:
            latest_row = latest_data.iloc[0] # Get the first row
            template_vars.update({
                "time_of_latest_reading": datetime.datetime.strftime(latest_row['datetime'], '%A at %H:%M'),
                "latest_temperature": round(latest_row.get('temperature', float('nan')), 1),
                "latest_humidity": round(latest_row.get('humidity', float('nan')), 1),
                "latest_rain_rate": round(latest_row.get('rain_rate', float('nan')) * 3600, 1),
                "latest_pressure": round(latest_row.get('pressure', float('nan')), 1),
                "latest_luminance": round(latest_row.get('luminance', float('nan')), 1),
                "latest_wind_speed_mph": round(latest_row.get('wind_speed', float('nan')) * 2.23694, 1),
                "latest_wind_direction": latest_row.get('wind_direction', "N/A"),
                "latest_wind_direction_converted": convert_wind_direction(latest_row.get('wind_direction')),
            })
            # Handle cases where a field might be N/A after rounding NaN
            for key in ["latest_temperature", "latest_humidity", "latest_rain_rate", "latest_pressure", "latest_luminance", "latest_wind_speed_mph"]:
                if pd.isna(template_vars[key]):
                    template_vars[key] = "N/A"

        except Exception as e:
            logging.error(f"Error processing latest data: {e}", exc_info=True)

    # Populate summary data
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

    # Generate annual summary table
    if not year_data.empty and 'datetime' in year_data.columns and not year_data['datetime'].empty:
        try:
            year_data_indexed = year_data.copy()
            year_data_indexed.set_index('datetime', inplace=True)
            
            # Ensure index is DatetimeIndex for groupby operations by month/day
            if not isinstance(year_data_indexed.index, pd.DatetimeIndex):
                 year_data_indexed.index = pd.to_datetime(year_data_indexed.index)

            total_rain = year_data_indexed.groupby(year_data_indexed.index.month)['rain'].sum()
            average_monthly_temperature = year_data_indexed.groupby(year_data_indexed.index.month)['temperature'].mean()
            max_monthly_temperature = year_data_indexed.groupby(year_data_indexed.index.month)['temperature'].max()
            min_monthly_temperature = year_data_indexed.groupby(year_data_indexed.index.month)['temperature'].min()

            monthly_data_summary = {
                'Av. Temp (C)': average_monthly_temperature,
                'Max Temp (C)': max_monthly_temperature,
                'Min Temp (C)': min_monthly_temperature,
                'Total Rain (mm)': total_rain,
            }

            df_summary = pd.DataFrame(monthly_data_summary).round(1)
            # Ensure all months 1-12 are present if needed, or handle missing months
            # df_summary = df_summary.reindex(range(1,13)) # If you want all months
            df_summary.index = [datetime.date(1900, month, 1).strftime('%B') for month in df_summary.index]
            template_vars["table"] = df_summary.to_html(classes='w3-table-all w3-responsive', escape=False, na_rep='N/A')

            # Calculate rainy days
            if 'rain' in year_data_indexed.columns:
                daily_rain_sum = year_data_indexed['rain'].resample('D').sum()
                num_rainy_days = (daily_rain_sum > 1.0).sum() # Days with more than 1mm of rain
                total_days = len(daily_rain_sum)
                if total_days > 0:
                    template_vars["num_rainy_days"] = num_rainy_days
                    template_vars["total_days"] = total_days
                    template_vars["rainy_percent"] = round(100 * (num_rainy_days / total_days))
                else:
                    template_vars["num_rainy_days"] = 0
                    template_vars["total_days"] = 0
                    template_vars["rainy_percent"] = 0
            else:
                logging.warning("Rain column not found for rainy day calculation.")


        except Exception as e:
            logging.error(f"Error generating annual table or rainy days: {e}", exc_info=True)

    return render_template("current.html", **template_vars)

# --- Plotting routes: Fetch data directly for each plot request ---

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

    if not pd.api.types.is_datetime64_any_dtype(last24h_data['datetime']):
        last24h_data['datetime'] = pd.to_datetime(last24h_data['datetime'], errors='coerce')
        last24h_data.dropna(subset=['datetime'], inplace=True)
        if last24h_data.empty:
             return Response("No valid datetime data for 24h rainfall plot.", mimetype='text/plain', status=204)

    rain_data = last24h_data.groupby(last24h_data['datetime'].dt.hour)['rain'].sum()
    # rain_data.index will be hours (0-23) that have data
    fig = plot_24h_bar_greyed(list(rain_data.index), list(rain_data.values), "Last 24 Hours Rainfall (by hour)", "Rainfall (mm)")
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
        # calplot expects a Series. `how='sum'` is okay here even for 0/1 data.
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

# If you were to run this locally for testing (Gunicorn will handle this in prod):
# if __name__ == "__main__":
#     # IMPORTANT: This is for local development only.
#     # Cloud Functions (Gen2) / Cloud Run will use Gunicorn via Procfile or auto-detection.
#     # Ensure Gunicorn is in requirements.txt for cloud deployment.
#     # PORT environment variable is set by Cloud Run.
#     port = int(os.environ.get("PORT", 8080))
#     # For local testing, you might want debug=True, but never in production.
#     # Gunicorn will not run this __main__ block.
#     app.run(host='0.0.0.0', port=port, debug=False) # Set debug=False for anything close to prod