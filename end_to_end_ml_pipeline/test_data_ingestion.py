import pandas as pd
from io import StringIO
import boto3
from data_ingestion import load_new_csvs

# SETUP

bucket_name = "my-bucket"
# this line creates an S3 client using boto3,
# which is Python’s library for interacting with AWS services.
# Normally, the client would connect to Amazon’s real S3 servers,
# but by specifying endpoint_url="https://localhost:4566",
# we are telling it to connect to a LocalStack instance running
# on our personal computer instead. LocalStack emulates AWS services locally,
# so this setup lets us create buckets, upload files, and test
# S3 operations without using a real AWS account or paying for cloud resources.
s3 = boto3.client(
    "s3",
    endpoint_url="http://localhost:4566",  # LocalStack endpoint
    aws_access_key_id="test",
    aws_secret_access_key="test")

# create a small dataframe
df = pd.DataFrame({
    "id": [0, 1, 2, 3],
    "value": [10, 20, 30, 40]
})

# a buffer is a temporary storage area
# in memory (RAM) where data can be written
# before it's read or sent somewhere else
# instead of writing the csv to a file on disk,
# here, we are writing it to a memory buffer
# an in-memory string object using StringIO

csv_buffer = StringIO()
# index = false means we're not including the df's
# index column in the csv
df.to_csv(csv_buffer, index=False)

# upload to s3 under "incoming/"
# s3.put_object(Bucket=bucket_name, Key="incoming/test_file.csv")
s3.put_object(
    Bucket=bucket_name,
    Key="incoming/test_file.csv",
    Body=csv_buffer.getvalue()
)

print("Uploaded sample CSV to incoming/")


# TEST

def test_load_new_csvs():
    files = load_new_csvs()

    for key, df in files:
        print(f"\nFile: {key}")
        print(df.head())

test_load_new_csvs()