# batch example

import boto3
import pandas as pd

s3 = boto3.client("s3", endpoint_url="http://localhost:4566")
s3.create_bucket(Bucket="my-bucket")

def load_new_csvs(prefix="incoming/"):
    # list all objects under prefix
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    if "contents" not in response:
        print("No new files found.")
        return []

    dfs = []
    for obj in response["contents"]:
        pass
    pass