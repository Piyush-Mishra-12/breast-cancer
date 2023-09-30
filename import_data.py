from sklearn.datasets import load_breast_cancer
from pymongo import MongoClient
import pandas as pd

# Reading our .csv file
ds = load_breast_cancer()
x = pd.DataFrame(ds.data, columns=ds.feature_names) # type: ignore
y = pd.Series(ds.target) # type: ignore
df = pd.concat([x, y], axis=1)
df.columns = list(ds.feature_names) + ['target'] # type: ignore

# Convert DataFrame to a list of dictionaries with column names as keys
data = df.to_dict(orient='records')

# Define the name of your Uniform Resource Identifier, database and collection
uri = "mongodb+srv://Piyush:cUxjKK4nwQVaHuK4@cluster0.opvarp6.mongodb.net/?retryWrites=true&w=majority"
database_name = "breast_cancer_data"
collection_name = "breast_cancer"

# Connect to MongoDB
client = MongoClient(uri)
db = client[database_name]
collection = db[collection_name]
collection.insert_many(data)
client.close() # Close the MongoDB connection