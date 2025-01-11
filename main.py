import os.path
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings
)

from llama_index.llms.openai import OpenAI
import os

from flask import Flask, stream_with_context

app = Flask(__name__)

@app.route("/streaming")
def streaming():

    # check if storage already exists
    PERSIST_DIR = "./storage"
    if not os.path.exists(PERSIST_DIR):
        # load the documents and create the index
        documents = SimpleDirectoryReader("./egobase").load_data()
        index = VectorStoreIndex.from_documents(documents)
        # store it for later
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        # load the existing index
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)

    # Either way we can now query the index
    Settings.llm = OpenAI(model="gpt-4o-mini")

    return stream_with_context(index.as_query_engine(streaming=True).query("How can i write the fibonacci function in php? come up with the code").response_gen)

