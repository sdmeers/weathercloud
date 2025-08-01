

import os
from datetime import datetime
from dateutil import tz
from google.cloud import firestore

# --- Configuration ---
# It is recommended to set the GOOGLE_APPLICATION_CREDENTIALS environment variable
# to point to your service account key file.
# Run "gcloud auth application-default login" to authenticate
# If you are running this on a GCP environment (like Cloud Shell or a VM),
# the client can often authenticate automatically without this.

# --- Firestore Client Initialization ---
try:
    # Initialize the Firestore client
    db = firestore.Client(database="weatherdata")
    print("Successfully connected to Firestore.")
except Exception as e:
    print(f"Error connecting to Firestore: {e}")
    print("Please ensure you have authenticated correctly.")
    db = None

def fix_spurious_rain_data(db_client):
    """
    Finds, displays, and corrects spurious rain and rain_rate data for a specific day
    after receiving user confirmation.
    """
    if not db_client:
        print("Firestore client is not available. Exiting.")
        return

    # --- Define the Target Date Range in Local Time ---
    # Define the target timezone for London, which correctly handles DST
    LONDON_TZ = tz.gettz('Europe/London')

    # Create timezone-aware datetime objects for the start and end of the day in London time.
    start_of_day_local = datetime(2025, 7, 30, 0, 0, 0, tzinfo=LONDON_TZ)
    end_of_day_local = datetime(2025, 7, 30, 23, 59, 59, 999999, tzinfo=LONDON_TZ)

    print(f"Searching for documents between {start_of_day_local.isoformat()} and {end_of_day_local.isoformat()} (London time)...")

    try:
        # --- Query for Documents on the Target Date ---
        # The Firestore client library will correctly handle the timezone-aware datetime objects.
        docs_query = db_client.collection('weather-readings').where(
            filter=firestore.FieldFilter('timestamp_UTC', '>=', start_of_day_local)
        ).where(
            filter=firestore.FieldFilter('timestamp_UTC', '<=', end_of_day_local)
        )

        # --- Stage 1: Fetch and Display Records for Review ---
        docs_to_update = list(docs_query.stream())

        if not docs_to_update:
            print("\nNo documents found for the specified date range. No changes are needed.")
            return

        print("\n--- Records to be Updated ---")
        print(f"Found {len(docs_to_update)} records for 30th July 2025. Current values:")
        for doc in docs_to_update:
            data = doc.to_dict()
            # Safely get rain and rain_rate with a default of 'N/A' if not present
            rain = data.get('rain', 'N/A')
            rain_rate = data.get('rain_rate', 'N/A')
            print(f"  - ID: {doc.id}, Rain: {rain}, Rain Rate: {rain_rate}")
        print("----------------------------")

        # --- Stage 2: Ask for User Confirmation ---
        confirmation = input("\nDo you want to proceed with setting 'rain' and 'rain_rate' to 0 for these records? (yes/no): ")

        if confirmation.lower() != 'yes':
            print("Update cancelled by user. No changes were made.")
            return

        # --- Stage 3: Perform the Update ---
        print("\nUser confirmed. Proceeding with update...")
        batch = db_client.batch()
        update_count = 0

        for doc in docs_to_update:
            doc_ref = doc.reference
            batch.update(doc_ref, {
                'rain': 0,
                'rain_rate': 0
            })
            update_count += 1

        batch.commit()
        print(f"\nSuccessfully updated {update_count} documents.")

    except Exception as e:
        print(f"An error occurred: {e}")

# --- Main Execution Block ---
if __name__ == "__main__":
    if db:
        print("\nThis script will find and update records from 30th July 2025.")
        print("It will first display the records and then ask for your confirmation before making any changes.")
        fix_spurious_rain_data(db)
