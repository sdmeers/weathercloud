import datetime
import html
import os
from urllib.parse import urlencode

import flask
import functions_framework
import pytz
from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter

RECORDS_PER_PAGE = 96

# Crawlers must not page through this table: every page is a billed Firestore
# query. In Nov 2026 Meta's crawler walked ?page=1..1203 under the old offset()
# pagination at ~56k reads per page (44M reads in one day).
ROBOTS_TXT = "User-agent: *\nDisallow: /\n"
NO_INDEX_HEADERS = {"X-Robots-Tag": "noindex, nofollow"}


def _serve_robots_txt():
    """Answer /robots.txt ourselves.

    functions-framework routes /robots.txt to a hard-coded 404 before the user
    function is called, so it cannot be handled inside display_weather_data().
    A before_request hook runs ahead of that routing decision.
    """
    if flask.request.path == "/robots.txt":
        return flask.Response(ROBOTS_TXT, mimetype="text/plain")
    return None


# functions-framework imports this module inside the Flask app context, which is
# the only point at which a hook can be registered on its app.
if flask.has_app_context():
    flask.current_app.before_request(_serve_robots_txt)


class InvalidCursor(ValueError):
    """A before=/after= value that is not an ISO-8601 timestamp."""


def parse_cursor(value):
    """Parse an ISO-8601 cursor into a UTC-aware datetime. Raises InvalidCursor."""
    try:
        dt = datetime.datetime.fromisoformat(value)
    except ValueError as e:
        raise InvalidCursor(value) from e
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.UTC)
    return dt


def format_cursor(timestamp):
    return timestamp.astimezone(datetime.UTC).isoformat()


def fetch_page(collection_ref, args):
    """Fetch one page of readings, newest first, using cursor pagination.

    Every branch reads at most RECORDS_PER_PAGE + 1 documents whatever page the
    caller is on; there is no offset() and no count().

      (no args)       newest page
      before=<ts>     the page of readings older than <ts>
      after=<ts>      the page of readings newer than <ts>
      oldest=1        the oldest page

    Returns (docs, has_newer, has_older), docs ordered newest first.
    """
    field = "timestamp_UTC"
    desc = firestore.Query.DESCENDING
    asc = firestore.Query.ASCENDING
    limit = RECORDS_PER_PAGE + 1  # one extra to learn whether another page exists

    if "before" in args:
        cursor = parse_cursor(args["before"])
        query = collection_ref.where(filter=FieldFilter(field, "<", cursor)).order_by(field, direction=desc)
        docs = list(query.limit(limit).stream())
        return docs[:RECORDS_PER_PAGE], True, len(docs) > RECORDS_PER_PAGE

    if "after" in args:
        cursor = parse_cursor(args["after"])
        query = collection_ref.where(filter=FieldFilter(field, ">", cursor)).order_by(field, direction=asc)
        docs = list(query.limit(limit).stream())
        return list(reversed(docs[:RECORDS_PER_PAGE])), len(docs) > RECORDS_PER_PAGE, True

    if "oldest" in args:
        docs = list(collection_ref.order_by(field, direction=asc).limit(limit).stream())
        return list(reversed(docs[:RECORDS_PER_PAGE])), len(docs) > RECORDS_PER_PAGE, False

    docs = list(collection_ref.order_by(field, direction=desc).limit(limit).stream())
    return docs[:RECORDS_PER_PAGE], False, len(docs) > RECORDS_PER_PAGE


def build_pagination_links(docs, has_newer, has_older):
    """Return the First/Previous/Next/Last buttons. Links are query-only relative
    URLs so they work on both the run.app and cloudfunctions.net hosts."""

    def button(query, label, extra_class="w3-margin-right"):
        href = html.escape("?" + urlencode(query), quote=True)
        return f'<a href="{href}" rel="nofollow" class="w3-button w3-dark-grey {extra_class}">{label}</a>'

    if not docs:  # cursor past either end of the data
        return button({}, "First Page")

    links = ""
    if has_newer:
        links += button({}, "First Page")
        links += button({"after": format_cursor(docs[0].get("timestamp_UTC"))}, "Previous Page")
    if has_older:
        links += button({"before": format_cursor(docs[-1].get("timestamp_UTC"))}, "Next Page")
        links += button({"oldest": "1"}, "Last Page", extra_class="")
    return links

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
    # The old ?page=N scheme is retired. Crawlers still hold ~1,200 such URLs, so
    # answer them without touching Firestore and mark them permanently gone.
    if "page" in request.args:
        return (
            '<p>This page no longer exists. <a href="?">View the latest readings</a>.</p>',
            410,
            {"Content-Type": "text/html", **NO_INDEX_HEADERS},
        )

    # Specify the Firestore database ID
    FIRESTORE_DATABASE_ID = "weatherdata"
    db = firestore.Client(database=FIRESTORE_DATABASE_ID)

    collection_ref = db.collection('weather-readings')

    try:
        docs, has_newer, has_older = fetch_page(collection_ref, request.args)
    except InvalidCursor:
        return "Invalid page cursor.", 400, {"Content-Type": "text/plain", **NO_INDEX_HEADERS}

    weather_data_rows = []
    # Define a timezone for conversion (e.g., Europe/London)
    # It's good practice to make this configurable via environment variables if deployed broadly.
    local_tz = pytz.timezone('Europe/London')

    for doc in docs:
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

        # 5) Retrieve signal strength (RSSI)
        signal_strength = data.get('signal_strength', 'N/A')


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
                <td>{signal_strength}</td>
            </tr>
        """)

    pagination_links = build_pagination_links(docs, has_newer, has_older)


    # Get URLs from environment variables, with fallbacks
    dashboard_URL = os.environ.get('DASHBOARD_URL', '/dashboard')
    chat_URL = os.environ.get('CHAT_URL', '/chat') 

    html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Weather Data Log</title>
            <meta name="robots" content="noindex, nofollow">
            <link rel="stylesheet" href="https://www.w3schools.com/w3css/4/w3.css">
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
            <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
            <script src="https://kit.fontawesome.com/e1d7788428.js" crossorigin="anonymous"></script>
            <style>
                html, body {{
                    font-family: 'Roboto', sans-serif;
                    font-size: 15px;
                    line-height: 1.5;
                    overflow-x: hidden;
                    background-color: white;
                }}
                .w3-main {{
                    padding-top: 44px; /* Account for fixed navbar */
                }}
                /* --- Start of new navbar CSS --- */
                .common_navbar {{
                    width: 100%;
                    overflow: hidden;
                    background-color: black;
                    color: white;
                    padding: 8px 16px;
                    z-index: 1001;
                    position: fixed;
                    top: 0;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    height: 44px;
                }}
                .hamburger-button {{ background: none; border: none; color: white; font-size: 22px; cursor: pointer; }}
                .navbar-title {{ font-size: 18px; font-weight: 400; }}
                .navbar-title i {{ margin-right: 8px; }}
                .sidenav {{
                    height: 100%;
                    width: 200px;
                    position: fixed;
                    z-index: 1002;
                    top: 0;
                    left: 0;
                    background-color: #111;
                    overflow-x: hidden;
                    transform: translateX(-100%);
                    transition: transform 0.3s ease-in-out;
                }}
                @media screen and (min-width: 600px) {{
                    .sidenav {{
                        width: 280px;
                    }}
                }}
                .sidenav-open {{ transform: translateX(0); }}
                .sidenav-header {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 10px 20px;
                    border-bottom: 1px solid #444;
                    min-height: 44px;
                }}
                .sidenav-title {{ color: white; font-size: 20px; margin: 0; font-weight: 500; }}
                .close-btn {{ background: none; border: none; color: #818181; font-size: 22px; cursor: pointer; }}
                .close-btn:hover {{ color: #f1f1f1; }}
                .sidenav a {{
                    padding: 10px 15px 10px 20px;
                    text-decoration: none;
                    font-size: 18px;
                    color: white;
                    display: block;
                    transition: 0.3s;
                    text-align: left;
                }}
                .sidenav a:hover {{ color: #f1f1f1; }}
                .sidenav .fa-fw {{ margin-right: 8px; }}
                /* --- End of new navbar CSS --- */
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; border-radius: 8px; overflow: hidden; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                tr:hover {{ background-color: #f1f1f1; }}
                .container {{ margin-top: 20px; padding: 20px; }}
                h1 {{ margin-bottom: 20px; color: #333; display: inline-block; vertical-align: middle; }}
                .pagination-top {{ margin-left: 20px; display: inline-block; vertical-align: middle; }}
                .pagination-bottom {{ margin-top: 20px; text-align: center; display: flex; justify-content: center; gap: 10px; }}
                .w3-button {{ border-radius: 8px; padding: 10px 20px; transition: background-color 0.3s ease; }}
                .w3-button:hover {{ background-color: #555 !important; }}
            </style>
        </head>
        <body>
            <!-- Sidenav/menu -->
            <nav class="sidenav" id="mySidenav">
              <div class="sidenav-header">
                <h4 class="sidenav-title">Navigation</h4>
                <button class="close-btn" id="close-sidenav-btn"><i class="fa-solid fa-xmark"></i></button>
              </div>
              <a href="https://weather-dashboard-728445650450.europe-west2.run.app/" class="w3-bar-item w3-button w3-padding"><i class="fa-solid fa-dashboard fa-fw"></i>  Summary</a>
              <a href="https://interactive-dashboard-728445650450.europe-west2.run.app/" class="w3-bar-item w3-button w3-padding"><i class="fa-solid fa-magnifying-glass-chart fa-fw"></i>  Dashboard</a>
              <a href="https://weather-chat-728445650450.europe-west2.run.app/" class="w3-bar-item w3-button w3-padding"><i class="fa-solid fa-comments fa-fw"></i>  Chatbot</a>
              <a href="https://display-weather-data-728445650450.europe-west2.run.app/" class="w3-bar-item w3-button w3-padding"><i class="fa-solid fa-database fa-fw"></i>  Data</a>
              <a href="https://europe-west1-weathercloud-460719.cloudfunctions.net/weather-image-classifier" class="w3-bar-item w3-button w3-padding"><i class="fa-solid fa-camera-retro fa-fw"></i>  Image Classifier</a>
            </nav>

            <!-- Top container -->
            <div class="common_navbar">
              <button class="hamburger-button" id="hamburger-btn">
                <i class="fa fa-bars"></i>
              </button>
              <span class="navbar-title"><i class="fa-solid fa-database fa-fw"></i> Weather Data</span>
            </div>

            <div class="w3-main">
                <div class="w3-container container">
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
                                    <th>Rain rate (mm/hr)</th>
                                    <th>Luminance (lux)</th>
                                    <th>Wind Speed (mph)</th>
                                    <th>Wind Direction</th>
                                    <th>Signal Strength (dBm)</th>
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
            </div>

            <script>
            document.addEventListener('DOMContentLoaded', function() {{
                const hamburgerBtn = document.getElementById('hamburger-btn');
                const sidenav = document.getElementById('mySidenav');
                const closeSidenavBtn = document.getElementById('close-sidenav-btn');

                hamburgerBtn.addEventListener('click', function() {{
                    sidenav.classList.add('sidenav-open');
                }});

                closeSidenavBtn.addEventListener('click', function() {{
                    sidenav.classList.remove('sidenav-open');
                }});

                document.addEventListener('click', function(event) {{
                    if (!sidenav.contains(event.target) && !hamburgerBtn.contains(event.target)) {{
                        sidenav.classList.remove('sidenav-open');
                    }}
                }});
            }});
            </script>
        </body>
        </html>
        """

    return html_content, 200, {'Content-Type': 'text/html', **NO_INDEX_HEADERS}