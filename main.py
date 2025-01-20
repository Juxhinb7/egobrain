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
import openai
from openai import OpenAIError

from flask import Flask, stream_with_context, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/streaming", methods=["POST"])
def streaming():
    content = request.get_json()
    os.environ['OPENAI_API_KEY'] = content["apiKey"]

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

    try:
        return stream_with_context(index.as_query_engine(streaming=True).query(content["prompt"]).response_gen)
    except OpenAIError as e:
        return jsonify({"errorMessage": e.args[0]})

