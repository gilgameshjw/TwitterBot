
import os
import json

from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index import StorageContext, load_index_from_storage


def build_db(db_local_path, db_local_dir):
    """build db from local path"""
    documents = SimpleDirectoryReader(db_local_path).load_data()
    print("building index... this can take several minutes...")
    index = VectorStoreIndex.from_documents(documents)
    # index.save(db_local_dir)
    index.storage_context.persist()
    return index.as_query_engine()

def load_db(db_local_dir):
    """load db from storage"""
    print("load db from storage...")
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context)
    return index.as_query_engine()

def search_utterance_in_db(query_engine, utterance):
    """Returns the most similar utterance from the database for the given utterance"""
    query = "Return a tuple: (similarity score, something very close to what was retweeted) to the most similar answer of the following tweet:\n" 
    query += utterance 
    r = query_engine.query(query).response
    print("--log:: search result: ", r)
    try:
        r = (float(r.split(",")[0][2:]), r.split(",")[1][:-1])
        if not r is None:
            if type(r) == tuple and len(r) == 2:
                return r[0], r[1]
    except Exception as e:
        pass

    return 0., ""