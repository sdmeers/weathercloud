import functions_framework
from google.cloud import firestore
import datetime
import pytz
import os

def convert_wind_direction(deg):
    """
    Converts a wind direction in degrees (0-360, where 0=SE, increasing clockwise)
    to a cardinal or intercardinal direction (N, NE, E, SE, S, SW, W, NW).
    
    This function assumes a specific mapping based on the provided degrees:
    0 = SE, 45 = S, 90 = SW, 135 = W, 180 = NW, 225 = N, 270 = NE, 315 = E.
    
    Args:
        deg (int or float): Wind direction in degrees.
        
    Returns:
        str: Cardinal or intercardinal direction string.
        
    Raises:
        ValueError: If the degree value does not match one of the predefined values.
    """
    # Ensure deg is a number, convert to int if float
    if not isinstance(deg, (int, float)):
        return "N/A" # Or raise a different error, or handle as unknown
    deg = int(deg)

    # Dictionary for mapping degrees to directions
    # Sorted for consistent lookup, or use if/elif chain
    direction_map = {
        225: "N",
        180: "NW",
        135: "W",
        90:  "SW",
        45:  "S",
        0:   "SE",
        315: "E",
        270: "NE"
    }
    
    # Simple direct lookup. If values are not exact, a range-based check would be needed.
    return direction_map.get(deg, "N/A") # Return N/A for invalid or unmapped degrees


@functions_framework.http
def display_weather_data(request):
    """
    Cloud Function to display raw weather readings from Firestore in a paginated table.
    """
    # Specify the Firestore database ID
    FIRESTORE_DATABASE_ID = "weatherdata"
    db = firestore.Client(database=FIRESTORE_DATABASE_ID)
    
    collection_ref = db.collection('weather-readings')

    # Get page number from request, default to 1
    page = int(request.args.get('page', 1))
    records_per_page = 96

    # Calculate offset for pagination
    # Note: Firestore's offset is costly for large datasets.
    # For robust pagination, consider using 'start_after' with the last document's timestamp.
    offset = (page - 1) * records_per_page

    # Query Firestore: order by timestamp_UTC descending
    base_query = collection_ref.order_by('timestamp_UTC', direction=firestore.Query.DESCENDING)

    # Get total document count for "Last Page" button.
    # This uses Firestore's aggregation queries.
    total_documents = 0
    try:
        agg_query = base_query.count()
        results = agg_query.get()
        if results:
            total_documents = results[0][0].value # Access the count value
    except Exception as e:
        print(f"Error getting total document count: {e}")
        # If count fails, last page button might not work or be accurate
        # Fallback to a value that allows some pagination, but might not be exact.
        # This prevents division by zero if total_documents remains 0 or calculation fails.
        total_documents = max(records_per_page * page, 1) # At least one page


    total_pages = (total_documents + records_per_page - 1) // records_per_page # Ceiling division

    # Apply offset and limit for pagination
    # Adding +1 to limit to check if there's a next page efficiently
    docs_stream = base_query.offset(offset).limit(records_per_page + 1).stream()
    docs = list(docs_stream)

    # Calculate has_next_page BEFORE generating pagination links
    has_next_page = len(docs) > records_per_page

    weather_data_rows = []
    # Define a timezone for conversion (e.g., Europe/London)
    # It's good practice to make this configurable via environment variables if deployed broadly.
    local_tz = pytz.timezone('Europe/London')

    # Iterate through documents for the current page only (up to records_per_page)
    for doc in docs[:records_per_page]:
        data = doc.to_dict()
        
        # Get timestamp_UTC. It should be a Firestore Timestamp object (which behaves like datetime).
        timestamp_utc = data.get('timestamp_UTC')

        if isinstance(timestamp_utc, datetime.datetime):
            # If it's already a datetime object (which it should be from Firestore)
            # Make sure it's timezone-aware (it usually is if stored properly by Firestore)
            # and convert it to the desired local timezone.
            if timestamp_utc.tzinfo is None:
                # If timezone naive, assume UTC and make it aware
                utc_dt = timestamp_utc.replace(tzinfo=pytz.utc)
            else:
                utc_dt = timestamp_utc # Already timezone-aware

            local_dt = utc_dt.astimezone(local_tz)
            local_timestamp = local_dt.strftime('%Y-%m-%d %H:%M:%S')
        else:
            local_timestamp = 'N/A' # Fallback if timestamp is missing or unexpected type

        # Retrieve and convert other data fields, providing 'N/A' as a default if missing
        temperature = data.get('temperature', 'N/A')
        pressure = data.get('pressure', 'N/A')
        humidity = data.get('humidity', 'N/A')
        rain = data.get('rain', 'N/A') # Keep original rain data

        # 1) Convert rain_rate from mm/s to mm/hr
        rain_rate_val = data.get('rain_rate')
        if isinstance(rain_rate_val, (int, float)):
            rain_rate = round(float(rain_rate_val) * 3600, 2) # Multiply by 3600 for mm/hr
        else:
            rain_rate = 'N/A'

        luminance = data.get('luminance', 'N/A')
        
        # 2) Convert wind_speed from m/s to mph
        wind_speed_val = data.get('wind_speed')
        if isinstance(wind_speed_val, (int, float)):
            wind_speed = round(float(wind_speed_val) * 2.23694, 2) # Multiply by 2.23694 for mph
        else:
            wind_speed = 'N/A'

        # 4) Convert wind_direction from degrees to compass points
        wind_direction_deg = data.get('wind_direction')
        wind_direction = convert_wind_direction(wind_direction_deg)


        weather_data_rows.append(f"""
            <tr>
                <td>{local_timestamp}</td>
                <td>{temperature}</td>
                <td>{pressure}</td>
                <td>{humidity}</td>
                <td>{rain}</td>
                <td>{rain_rate}</td>
                <td>{luminance}</td>
                <td>{wind_speed}</td>
                <td>{wind_direction}</td>
            </tr>
        """)

    # Generate pagination links
    pagination_links = ""
    # "First Page" button
    if page > 1:
        pagination_links += f'<a href="?page=1" class="w3-button w3-dark-grey w3-margin-right">First Page</a>'
    
    # "Previous Page" button
    if page > 1:
        pagination_links += f'<a href="?page={page - 1}" class="w3-button w3-dark-grey w3-margin-right">Previous Page</a>'
    
    # "Next Page" button
    if has_next_page: # Now correctly defined
        pagination_links += f'<a href="?page={page + 1}" class="w3-button w3-dark-grey w3-margin-right">Next Page</a>'
    
    # "Last Page" button
    if page < total_pages:
        pagination_links += f'<a href="?page={total_pages}" class="w3-button w3-dark-grey">Last Page</a>'


    # Get URLs from environment variables, with fallbacks
    dashboard_URL = os.environ.get('DASHBOARD_URL', '/dashboard')
    chat_URL = os.environ.get('CHAT_URL', '/chat') 

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Raspberry Pi Weather Log</title>
        <link rel="stylesheet" href="https://www.w3schools.com/w3css/4/w3.css">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
        <script src="https://kit.fontawesome.com/e1d7788428.js" crossorigin="anonymous"></script>
        <style>
            body {{ background-color: white; font-family: "Inter", sans-serif; }}
            .w3-top {{ position: sticky; top: 0; width: 100%; }}
            .w3-bar-item a {{ text-decoration: none; color: inherit; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; border-radius: 8px; overflow: hidden; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f2f2f2; font-weight: bold; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            tr:hover {{ background-color: #f1f1f1; }}
            .container {{ margin-top: 60px; padding: 20px; }}
            h1 {{ margin-bottom: 20px; color: #333; display: inline-block; vertical-align: middle; }}
            .pagination-top {{ margin-left: 20px; display: inline-block; vertical-align: middle; }}
            .pagination-bottom {{ margin-top: 20px; text-align: center; display: flex; justify-content: center; gap: 10px; }}
            .w3-button {{ border-radius: 8px; padding: 10px 20px; transition: background-color 0.3s ease; }}
            .w3-button:hover {{ background-color: #555 !important; }}
        </style>
    </head>
    <body class="w3-white">

        <!-- Top container -->
        <div class="w3-bar w3-top w3-black w3-large" style="z-index:4; border-radius: 0 0 8px 8px;">
            <span class="w3-bar-item w3-left"><i class="fa-solid fa-magnifying-glass-chart"></i> <a href="{dashboard_URL}">Weather Summary</a></span>
            <span class="w3-bar-item w3-right"><i class="fa-solid fa-comments"></i> <a href="{chat_URL}">Weather Chat</a></span>
        </div>

        <div class="w3-container w3-margin-top container">
            <h1>Raspberry Pi Weather Log</h1>
            <div class="pagination-top">
                {pagination_links}
            </div>

            <div class="w3-responsive w3-card-4" style="border-radius: 8px;">
                <table class="w3-table w3-striped w3-bordered">
                    <thead>
                        <tr>
                            <th>Timestamp (local)</th>
                            <th>Temperature (C)</th>
                            <th>Pressure (hPa)</th>
                            <th>Humidity (%)</th>
                            <th>Rain (mm)</th>
                            <th>Rain rate (mm/hr)</th> <!-- Updated label -->
                            <th>Luminance (lux)</th>
                            <th>Wind Speed (mph)</th> <!-- Updated label -->
                            <th>Wind Direction</th> <!-- Updated label -->
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(weather_data_rows)}
                    </tbody>
                </table>
            </div>

            <div class="pagination-bottom">
                {pagination_links}
            </div>
        </div>

    </body>
    </html>
    """
    return html_content, 200, {'Content-Type': 'text/html'}