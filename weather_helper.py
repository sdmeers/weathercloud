from google.cloud import firestore
import pandas as pd
from datetime import datetime, timedelta
from dateutil import tz 
import logging

# It's good practice to initialize the client once if this module is part of a larger app.
# If this helper is used in different contexts (e.g., multiple Cloud Functions),
# passing the db client as an argument is more flexible.
# db = firestore.Client(database="weatherdata") # Or your specific DB ID

LONDON_TZ = tz.gettz('Europe/London')

def get_data(db_client: firestore.Client, time_range="latest"):
    """
    Retrieves weather data from Firestore, mirroring the behavior of the original
    SQL-based get_data function.

    Args:
        db_client: An initialized Firestore client instance.
        time_range (str or tuple): Defines the period for data retrieval.
            "latest", "today", "yesterday", "week", "month", "year", "all",
            ('ud', 'YYYY-MM-DD'), ('udm', 'YYYY-MM').

    Returns:
        pandas.DataFrame: A DataFrame containing the weather data, or an empty
                          DataFrame if no data is found or an error occurs.
    """
    now_local = datetime.now(LONDON_TZ)
    collection_ref = db_client.collection('weatherdata')
    query = None

    try:
        if time_range == "latest":
            query = collection_ref.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(1)
        
        elif time_range == "today":
            day_str = now_local.strftime("%j")
            year_str = now_local.strftime("%Y")
            query = collection_ref.where('day', '==', day_str) \
                                  .where('year', '==', year_str) \
                                  .order_by('timestamp', direction=firestore.Query.DESCENDING)
        
        elif time_range == "yesterday":
            yesterday_local = now_local - timedelta(days=1)
            day_str = yesterday_local.strftime("%j")
            year_str = yesterday_local.strftime("%Y")
            query = collection_ref.where('day', '==', day_str) \
                                  .where('year', '==', year_str) \
                                  .order_by('timestamp', direction=firestore.Query.DESCENDING)
        
        elif time_range == "week": # Current week (Mon-Sun) based on %W
            week_str = now_local.strftime("%W")
            year_str = now_local.strftime("%Y")
            query = collection_ref.where('week', '==', week_str) \
                                  .where('year', '==', year_str) \
                                  .order_by('timestamp', direction=firestore.Query.DESCENDING)
        
        elif time_range == "month":
            month_str = now_local.strftime("%m")
            year_str = now_local.strftime("%Y")
            query = collection_ref.where('month', '==', month_str) \
                                  .where('year', '==', year_str) \
                                  .order_by('timestamp', direction=firestore.Query.DESCENDING)
        
        elif time_range == "year":
            year_str = now_local.strftime("%Y")
            query = collection_ref.where('year', '==', year_str) \
                                  .order_by('timestamp', direction=firestore.Query.DESCENDING)
        
        elif time_range == "all":
            query = collection_ref.order_by('timestamp', direction=firestore.Query.DESCENDING)
        
        elif isinstance(time_range, tuple) and len(time_range) == 2:
            if time_range[0] == 'ud':  # User-defined day e.g., ('ud', '2024-05-23')
                date_obj_local = datetime.strptime(time_range[1], "%Y-%m-%d").replace(tzinfo=LONDON_TZ)
                day_str = date_obj_local.strftime("%j")
                year_str = date_obj_local.strftime("%Y")
                query = collection_ref.where('day', '==', day_str) \
                                      .where('year', '==', year_str) \
                                      .order_by('timestamp', direction=firestore.Query.DESCENDING)
            elif time_range[0] == 'udm':  # User-defined month e.g., ('udm', '2024-05')
                date_obj_local = datetime.strptime(time_range[1], "%Y-%m").replace(tzinfo=LONDON_TZ)
                month_str = date_obj_local.strftime("%m")
                year_str = date_obj_local.strftime("%Y")
                query = collection_ref.where('month', '==', month_str) \
                                      .where('year', '==', year_str) \
                                      .order_by('timestamp', direction=firestore.Query.DESCENDING)
            else:
                logging.error(f"Unknown user-defined time_range tuple: {time_range}")
                return pd.DataFrame()
        else:
            logging.error(f"Unknown time_range: {time_range}")
            return pd.DataFrame()

        docs = query.stream()
        data_list = [doc.to_dict() for doc in docs]

        if not data_list:
            return pd.DataFrame()

        df = pd.DataFrame(data_list)

        # Timestamp conversion to match original behavior (stored as UTC string in Firestore)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp']) # Parses ISO string, typically to UTC-aware
            if df['timestamp'].dt.tz is None: # If somehow it became naive
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
            df['timestamp'] = df['timestamp'].dt.tz_convert(LONDON_TZ)
        
        # Ensure essential columns exist (matching the original SQL table structure if possible)
        # Add more columns if they were part of your original SELECT * and you expect them
        expected_cols = ["temperature", "pressure", "humidity", "rain", "rain_rate", 
                         "luminance", "wind_speed", "wind_direction", 
                         "day", "week", "month", "year", "timestamp"] 
        for col in expected_cols:
            if col not in df.columns:
                df[col] = pd.NA # Use pandas NA for missing data

        return df

    except Exception as e:
        logging.error(f"Error fetching data from Firestore for time_range '{time_range}': {e}")
        # Firestore might raise errors if indexes are missing. Check logs for links to create them.
        return pd.DataFrame()

# Example Usage (if you run this file directly for testing):
if __name__ == '__main__':
    # This assumes you have GOOGLE_APPLICATION_CREDENTIALS set for local testing
    # or are running in a GCP environment with default credentials.
    
    # --- IMPORTANT ---
    # For this example to run, you MUST have data in Firestore that includes
    # the 'day', 'week', 'month', 'year' fields.
    # If not, queries for "today", "week", etc., will return empty.
    # "latest" and "all" should work on existing data but might lack these date part columns.
    # -------------------

    logging.basicConfig(level=logging.INFO)
    
    # Initialize client for testing
    # Replace "your-gcp-project-id" if not using default from environment
    # Replace "weatherdata" if your Firestore database ID is different
    try:
        test_db_client = firestore.Client(database="weatherdata") # Adjust database name if needed
        
        logging.info("Fetching latest record:")
        latest_df = get_data_firestore(test_db_client, "latest")
        print(latest_df)

        logging.info("\nFetching today's records:")
        today_df = get_data_firestore(test_db_client, "today")
        print(today_df)
        
        # logging.info("\nFetching records for week:")
        # week_df = get_data_firestore(test_db_client, "week")
        # print(week_df)

        # logging.info("\nFetching records for user-defined day ('ud', 'YYYY-MM-DD'):")
        # You'd need to ensure data exists for this specific day with the new fields
        # specific_day_df = get_data_firestore(test_db_client, ('ud', '2024-05-20'))
        # print(specific_day_df)

    except Exception as e:
        logging.error(f"Error during Firestore test: {e}")
        logging.error("Ensure GOOGLE_APPLICATION_CREDENTIALS are set for local testing,")
        logging.error("or that you are running in a GCP environment with appropriate permissions.")
        logging.error("Also ensure your Firestore database 'weatherdata' (or as specified) exists and has data.")