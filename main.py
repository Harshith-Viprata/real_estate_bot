from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("API_KEY")
MODEL = os.getenv("MODEL")

if not API_KEY or not MODEL:
    raise ValueError("API_KEY or MODEL is missing from the environment variables.")

# Initialize the LLM (ChatGroq)
llm = ChatGroq(api_key=API_KEY, model=MODEL)

# Initialize the embedding model for document and query embedding
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Define the domain context for the AI assistant
domain_context = """
You are a real estate AI assistant. Your knowledge covers the real estate market, property values, investment strategies, home buying and selling processes, mortgages, and other related real estate topics. Please provide informative, helpful, and clear responses based only on real estate topics. Do not provide information outside the real estate domain.
"""

# Function to query the vector database for relevant documents
def query_vector_db(prompt, collection):
    relevant_docs = collection.query(query_texts=[prompt], n_results=3)
    return [str(doc) for doc in relevant_docs['documents']]

# Function to generate a response from the LLM
def get_llm_response(prompt, relevant_docs):
    full_prompt = domain_context + " Here are some relevant documents:\n" + "\n".join(relevant_docs) + "\n" + prompt
    response = llm.invoke([HumanMessage(content=full_prompt)], max_tokens=400, temperature=0.7)
    return response.content
