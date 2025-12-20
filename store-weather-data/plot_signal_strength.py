import os
import datetime
import logging
from google.cloud import firestore
import matplotlib.pyplot as plt
import pandas as pd
from dateutil import tz

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
FIRESTORE_DATABASE_ID = "weatherdata"
PROJECT_ID = "weathercloud-460719"
COLLECTION_NAME = "weather-readings"
DAYS_TO_PLOT = 7

def get_firestore_client():
    """Initializes and returns a Firestore client."""
    try:
        # Explicitly specify the database ID and project ID
        db = firestore.Client(project=PROJECT_ID, database=FIRESTORE_DATABASE_ID)
        return db
    except Exception as e:
        logging.critical(f"Failed to initialize Firestore client: {e}")
        return None

def fetch_data(db, days):
    """Fetches weather data for the last 'days' days."""
    end_time = datetime.datetime.now(datetime.timezone.utc)
    start_time = end_time - datetime.timedelta(days=days)
    
    logging.info(f"Fetching data from {start_time} to {end_time}...")
    
    collection_ref = db.collection(COLLECTION_NAME)
    # Use FieldFilter to avoid warning about positional arguments
    from google.cloud.firestore_v1.base_query import FieldFilter
    query = collection_ref.where(filter=FieldFilter("timestamp_UTC", ">=", start_time)).order_by("timestamp_UTC")
    
    docs = query.stream()
    
    data = []
    for doc in docs:
        doc_dict = doc.to_dict()
        if 'signal_strength' in doc_dict and 'timestamp_UTC' in doc_dict:
             data.append({
                'timestamp': doc_dict['timestamp_UTC'],
                'signal_strength': doc_dict['signal_strength']
            })
    
    logging.info(f"Retrieved {len(data)} records with signal strength data.")
    return data

def plot_data(data):
    """Plots the signal strength data."""
    if not data:
        logging.warning("No data to plot.")
        return

    df = pd.DataFrame(data)
    
    # Ensure timestamp is datetime aware (it should be coming from Firestore)
    # Convert to local time for better readability if desired, but keeping UTC for consistency with server
    # Let's convert to local time (assuming system local) for the plot
    df['timestamp'] = df['timestamp'].dt.tz_convert(tz.tzlocal())

    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['signal_strength'], marker='o', linestyle='-', markersize=2, alpha=0.7)
    
    plt.title(f'WiFi Signal Strength (Last {DAYS_TO_PLOT} Days)')
    plt.xlabel('Time')
    plt.ylabel('Signal Strength (dBm)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    # Save the plot
    output_file = 'signal_strength.png'
    plt.savefig(output_file)
    logging.info(f"Plot saved to {output_file}")
    
    # Show the plot
    plt.show()

def main():
    db = get_firestore_client()
    if db:
        data = fetch_data(db, DAYS_TO_PLOT)
        plot_data(data)

if __name__ == "__main__":
    main()
