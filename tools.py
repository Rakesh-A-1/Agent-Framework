from crewai.tools import tool
from decouple import config
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import requests

# @tool("fetch_api_data")
# def fetch_from_api(query: str):
#     """Fetch product data from the DummyJSON API."""
#     response = requests.get("https://dummyjson.com/products")
#     products = response.json().get("products", [])
#     return [
#         p for p in products
#         if query.lower() in p.get("title", "").lower()
#         or query.lower() in p.get("brand", "").lower()
#     ]

@tool("fetch_api_data")
def fetch_from_api(query: str):
    """Fetch product data from the DummyJSON API with optional filtering."""
    response = requests.get("https://dummyjson.com/products")
    products = response.json().get("products", [])

    # Check if the query is a generic "all products" type
    if query.lower() in ["all products", "show me all products", "give all products"]:
        return products

    # Otherwise, filter by title or brand
    filtered = [
        p for p in products
        if query.lower() in p.get("title", "").lower()
        or query.lower() in p.get("brand", "").lower()
    ]
    return filtered

@tool("search_pinecone")
def search_pinecone(query: str):
    """Search semantically similar products from Pinecone."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    pc = Pinecone(api_key=config("PC_API_KEY"))
    index = pc.Index("ecommerce-products")
    query_emb = model.encode(query).tolist()
    results = index.query(vector=query_emb, top_k=20, include_metadata=True)
    return [m["metadata"] for m in results["matches"]]
