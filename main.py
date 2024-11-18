# from langchain_groq import ChatGroq
# from langchain.schema import HumanMessage
# from sentence_transformers import SentenceTransformer
# from chromadb import Client
# from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
# import requests
# import datetime

# # Initialize the LLM (ChatGroq)
# llm = ChatGroq(api_key="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", model="llama3-70b-8192")

# # Initialize the embedding model for document and query embedding
# embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# # Initialize the ChromaDB client for retrieving documents
# client = Client()

# # Define domain context
# domain_context = """
# You are a real estate AI assistant. Your knowledge covers the real estate market, property values, investment strategies, home buying and selling processes, mortgages, and other related real estate topics. Please provide informative, helpful, and clear responses based only on real estate topics. Do not provide information outside the real estate domain.
# """

# # Function to query the vector database for relevant documents
# def query_vector_db(query, n_results=3):
#     collection_name = "real_estate_docs"  # Use a predefined collection name for retrieval only
#     collection = client.get_or_create_collection(
#         name=collection_name,
#         embedding_function=SentenceTransformerEmbeddingFunction('all-MiniLM-L6-v2')
#     )
#     results = collection.query(query_texts=[query], n_results=n_results)
#     return results['documents'] if results['documents'] else ["No relevant documents found."]

# # Main loop for prompting and response
# while True:
#     # Input prompt from the user
#     prompt = input("Enter your real estate query (or type 'exit' to quit): ")
    
#     if prompt.lower() == 'exit':
#         print("Exiting the program.")
#         break

#     # Print the user's prompt
#     print("User prompt:", prompt)
    
#     # Query the vector database for relevant documents
#     relevant_docs = query_vector_db(prompt)
#     relevant_docs_str = [str(doc) for doc in relevant_docs]
#     print("Relevant documents:", relevant_docs_str)

#     # Construct the full prompt
#     full_prompt = domain_context + " Here are some relevant documents:\n" + "\n".join(relevant_docs_str) + "\n" + prompt
#     print("Full prompt sent to LLM:", full_prompt)

#     # Generate the response from the LLM
#     response = llm.invoke([HumanMessage(content=full_prompt)], max_tokens=400, temperature=0.7)

#     # Print the assistant's response
#     print("Assistant response:", response.content)


from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from sentence_transformers import SentenceTransformer

# Initialize the LLM (ChatGroq)
llm = ChatGroq(api_key="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", model="llama3-70b-8192")

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
