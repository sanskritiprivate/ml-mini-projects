# batch example

import boto3
import pandas as pd

s3 = boto3.client("s3")
bucket_name = "my-data-bucket"

def load_new_csvs(prefix="incoming/"):
    pass