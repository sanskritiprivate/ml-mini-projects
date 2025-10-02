# batch example

import boto3
import pandas as pd

s3 = boto3.client(
    "s3",
    endpoint_url="http://localhost:4566",  # LocalStack endpoint
    aws_access_key_id="test",
    aws_secret_access_key="test")
bucket_name = "my-bucket"
s3.create_bucket(Bucket=bucket_name)

def load_new_csvs(prefix="incoming/"):
    # list all objects under prefix
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    if "Contents" not in response:
        print("No new files found.")
        return []

    dfs = []
    for obj in response["Contents"]:
        key = obj["Key"]
        if key.endswith(".csv"):
            print(f"loading {key} ...")
            # download file object
            file_obj = s3.get_object(Bucket=bucket_name, Key=key)
            df = pd.read_csv(file_obj["Body"])
            dfs.append((key, df))
    return dfs


# stream example