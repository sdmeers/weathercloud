from google.cloud import firestore

# Initialize Firestore client
# Make sure your GOOGLE_APPLICATION_CREDENTIALS environment variable is set
# or you've configured authentication in another way.
db = firestore.Client(database="weatherdata")  # Specify the database name

# --- Configuration ---
collection_name = 'weather-readings'
old_field_name = 'timestamp_UTC'
new_field_name = 'timestamp'
batch_size = 500  # Number of documents to update in a single batch write

def rename_firestore_field(collection_ref, old_field, new_field, batch_size):
    """
    Renames a field in all documents within a specified Firestore collection.
    Uses batched writes for efficiency.
    """
    print(f"Starting field rename from '{old_field}' to '{new_field}' in collection '{collection_ref.id}'...")

    docs_to_process = collection_ref.stream()
    processed_count = 0
    batch = db.batch()

    for doc in docs_to_process:
        doc_data = doc.to_dict()

        if old_field in doc_data:
            # Get the value of the old field
            field_value = doc_data[old_field]

            # Add the new field with the old field's value
            batch.update(doc.reference, {
                new_field: field_value,
                old_field: firestore.DELETE_FIELD  # Delete the old field
            })
            processed_count += 1

            if processed_count % batch_size == 0:
                print(f"Committing batch of {batch_size} documents. Total processed: {processed_count}")
                try:
                    batch.commit()
                    batch = db.batch()  # Start a new batch
                except Exception as e:
                    print(f"Error committing batch: {e}")
                    # You might want to log these errors and potentially retry or handle them
                    # For simplicity, this example just prints, but in production,
                    # robust error handling is crucial.
                    batch = db.batch() # Reset batch to avoid issues

    # Commit any remaining documents in the last batch
    if processed_count % batch_size != 0:
        print(f"Committing final batch of {processed_count % batch_size} documents. Total processed: {processed_count}")
        try:
            batch.commit()
        except Exception as e:
            print(f"Error committing final batch: {e}")

    print(f"Finished renaming field. Total documents processed: {processed_count}")

if __name__ == "__main__":
    collection_ref = db.collection(collection_name)
    rename_firestore_field(collection_ref, old_field_name, new_field_name, batch_size)