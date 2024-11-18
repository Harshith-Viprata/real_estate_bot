from flask import Flask, request, jsonify
from flask_cors import CORS
from main import query_vector_db, get_llm_response, domain_context, llm
from chromadb import Client
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

app = Flask(__name__)
CORS(app)  # Allow requests from any origin

# Initialize ChromaDB client and collection for storing documents
client = Client()
embedding_function = SentenceTransformerEmbeddingFunction('all-MiniLM-L6-v2')
collection = client.get_or_create_collection(
    name="real_estate_docs",
    embedding_function=embedding_function
)

# Endpoint to receive and process a prompt (POST)
@app.route('/submit_prompt', methods=['POST'])
def submit_prompt():
    try:
        data = request.get_json()
        prompt = data.get('prompt')

        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400

        relevant_docs = query_vector_db(prompt, collection)
        response = get_llm_response(prompt, relevant_docs)

        return jsonify({'response': response}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Endpoint to process a prompt using GET request
@app.route('/get_response', methods=['GET'])
def get_response():
    try:
        prompt = request.args.get('prompt')

        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400

        relevant_docs = query_vector_db(prompt, collection)
        response = get_llm_response(prompt, relevant_docs)

        return jsonify({'response': response}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)