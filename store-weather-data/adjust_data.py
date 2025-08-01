import os
from datetime import datetime
from dateutil import tz
from google.cloud import firestore

# --- User Configuration ---
# Instructions: Modify the values in this section to control the script's behavior.

# 1. Set the date and time range for the update.
# Format: 'YYYY-MM-DD HH:MM' or 'YYYY-MM-DD'. The range is inclusive.
START_DATE_STR = '2025-07-30 00:00'
END_DATE_STR = '2025-07-30 23:59'

# 2. List the parameters (fields) you want to adjust in the database.
# Example: ['rain', 'rain_rate'] or ['temperature']
PARAMETERS_TO_ADJUST = ['rain', 'rain_rate']

# 3. Choose the operation to perform.
# Options: 'SET', 'MODIFY'
#   - 'SET': Sets the parameter to the exact value in OPERATION_VALUE.
#   - 'MODIFY': Adds the OPERATION_VALUE to the existing parameter value (use a negative number to subtract).
OPERATION_TYPE = 'SET'

# 4. Set the value to be used in the operation.
# For 'SET', this is the new value. For 'MODIFY', this is the amount to add/subtract.
OPERATION_VALUE = 0

# --- End of User Configuration ---


# --- Firestore Client Initialization ---
try:
    db = firestore.Client(database="weatherdata")
    print("Successfully connected to Firestore.")
except Exception as e:
    print(f"Error connecting to Firestore: {e}")
    print("Please ensure you have authenticated correctly with 'gcloud auth application-default login'.")
    db = None

def parse_datetime(date_str):
    """Parses a string that could be a date or a datetime."""
    try:
        return datetime.strptime(date_str, '%Y-%m-%d %H:%M')
    except ValueError:
        return datetime.strptime(date_str, '%Y-%m-%d')

def adjust_data(db_client):
    """
    Finds, displays, and adjusts data for a given date range and parameters
    after receiving user confirmation.
    """
    if not db_client:
        print("Firestore client is not available. Exiting.")
        return

    # --- Define the Target Date Range in Local Time ---
    try:
        LONDON_TZ = tz.gettz('Europe/London')
        start_date = parse_datetime(START_DATE_STR).replace(tzinfo=LONDON_TZ)
        end_date = parse_datetime(END_DATE_STR).replace(tzinfo=LONDON_TZ)
        # If user only provides a date for the end, set it to the end of that day
        if len(END_DATE_STR.split()) == 1:
            end_date = end_date.replace(hour=23, minute=59, second=59, microsecond=999999)

    except ValueError:
        print(f"Error: Invalid date format. Please use 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM'.")
        return

    print(f"Searching for documents between {start_date.isoformat()} and {end_date.isoformat()} (London time)...")

    try:
        # --- Query for Documents on the Target Date ---
        docs_query = db_client.collection('weather-readings').where(
            filter=firestore.FieldFilter('timestamp_UTC', '>=', start_date)
        ).where(
            filter=firestore.FieldFilter('timestamp_UTC', '<=', end_date)
        )

        # --- Stage 1: Fetch and Display Records for Review ---
        docs_to_update = list(docs_query.stream())

        if not docs_to_update:
            print("\nNo documents found for the specified date range. No changes are needed.")
            return

        print("\n--- Records to be Updated ---")
        print(f"Found {len(docs_to_update)} records. The following changes will be made:")
        for doc in docs_to_update:
            data = doc.to_dict()
            print(f"  - ID: {doc.id}")
            for param in PARAMETERS_TO_ADJUST:
                current_value = data.get(param)
                adjusted_value = 'N/A'

                if current_value is not None and isinstance(current_value, (int, float)):
                    if OPERATION_TYPE == 'SET':
                        adjusted_value = OPERATION_VALUE
                    elif OPERATION_TYPE == 'MODIFY':
                        adjusted_value = current_value + OPERATION_VALUE
                    print(f"    - {param}: Current: {current_value:.4f}, New: {adjusted_value:.4f}")
                else:
                    print(f"    - {param}: Current: {current_value}, New: (Cannot be calculated)")
        print("----------------------------")

        # --- Stage 2: Ask for User Confirmation ---
        print(f"\nSummary of intended operation:")
        print(f"  - Operation: {OPERATION_TYPE}")
        print(f"  - Parameters: {PARAMETERS_TO_ADJUST}")
        print(f"  - Value: {OPERATION_VALUE}")
        confirmation = input("\nDo you want to proceed with this operation? (yes/no): ")

        if confirmation.lower() != 'yes':
            print("Update cancelled by user. No changes were made.")
            return

        # --- Stage 3: Perform the Update ---
        print("\nUser confirmed. Proceeding with update...")
        batch = db_client.batch()
        update_payload = {}

        if OPERATION_TYPE == 'SET':
            for param in PARAMETERS_TO_ADJUST:
                update_payload[param] = OPERATION_VALUE
        elif OPERATION_TYPE == 'MODIFY':
            for param in PARAMETERS_TO_ADJUST:
                update_payload[param] = firestore.Increment(OPERATION_VALUE)
        else:
            print(f"Error: Invalid OPERATION_TYPE '{OPERATION_TYPE}'. No changes were made.")
            return

        for doc in docs_to_update:
            batch.update(doc.reference, update_payload)

        batch.commit()
        print(f"\nSuccessfully updated {len(docs_to_update)} documents.")

    except Exception as e:
        print(f"An error occurred: {e}")

# --- Main Execution Block ---
if __name__ == "__main__":
    if db:
        print("\n--- General Purpose Firestore Data Adjustment Script ---")
        print("This script will modify data based on the configuration at the top of the file.")
        print("Please review the configuration carefully before proceeding.")
        adjust_data(db)