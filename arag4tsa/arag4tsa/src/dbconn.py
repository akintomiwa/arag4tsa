# import pymongo
# # from dotenv import load_dotenv
# # import os
# # from google.colab import userdata

# # load_dotenv()

# from ignore import MONGO_URI


# def get_mongo_client(mongo_uri):
#     """Establish connection to the MongoDB."""
#     try:
#         client = pymongo.MongoClient(mongo_uri)
#         print("Connection to MongoDB successful")
#         return client
#     except pymongo.errors.ConnectionFailure as e:
#         print(f"Connection failed: {e}")
#         return None


# # Get the MongoDB URI from environment variables
# # mongo_uri = os.getenv("MONGO_URI")
# mongo_uri = MONGO_URI
# if not mongo_uri:
#     print("MONGO_URI not set in environment variables")


# # mongo_uri = userdata.get("MONGO_URI")
# # if not mongo_uri:
# #     print("MONGO_URI not set in environment variables")

# mongo_client = get_mongo_client(mongo_uri)

# # Ingest data into MongoDB
# db = mongo_client["movies"]
# collection = db["movie_collection_2"]

# print(f"DB: {db}")
# print(f"Collection: {collection}")

# # Delete any existing records in the collection
# collection.delete_many({})

# documents = dataset_df.to_dict("records")
# collection.insert_many(documents)

# print("Data ingestion into MongoDB completed")