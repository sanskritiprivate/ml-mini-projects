import pandas as pd
from io import StringIO

# create a small dataframe
df = pd.DataFrame({
    "id": [0,1,2,3],
    "value": [10,20,30,40]
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