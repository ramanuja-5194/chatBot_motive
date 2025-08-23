import boto3
import os
from dotenv import load_dotenv

load_dotenv()  # Load AWS credentials from .env

def download_s3_file(bucket_name: str, key_name: str, local_filename: str):
    """
    Downloads a file from S3 and saves it locally.
    """
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_REGION", "us-east-1")

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region
    )

    s3_client.download_file(bucket_name, key_name, local_filename)


if __name__ == "__main__":
    bucket_name = "motiverse-2025-data"
    key_name = "web_content.txt"
    local_filename = "web_content.txt"

    download_s3_file(bucket_name, key_name, local_filename)
    print(f"✅ File saved at {os.path.abspath(local_filename)}")
