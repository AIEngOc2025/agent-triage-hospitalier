import os

from google.cloud import storage


def download_blob(bucket_name, source_blob_prefix, destination_folder):
    """Downloads a blob/folder from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=source_blob_prefix)

    for blob in blobs:
        # Create local directory structure
        relative_path = os.path.relpath(blob.name, source_blob_prefix)
        local_file_path = os.path.join(destination_folder, relative_path)
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        if not blob.name.endswith("/"):
            print(f"📥 Downloading {blob.name} to {local_file_path}")
            blob.download_to_filename(local_file_path)


if __name__ == "__main__":
    bucket_name = "modeles-triage-hospitalier"
    source_prefix = "merged_dpo_final_chsa"
    destination = "/app/models/merged_dpo_final_chsa"

    print(f"🚀 Starting download from gs://{bucket_name}/{source_prefix}...")
    download_blob(bucket_name, source_prefix, destination)
    print("✅ Model download complete.")
