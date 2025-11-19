from crewai.tools import tool
from decouple import config
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import requests

@tool
def fetch_from_api(query: dict) -> list:
    """
    Fetch all products from DummyJSON API.
    Do NOT manually filter here — agent handles filtering/verification.
    """
    try:
        response = requests.get("https://dummyjson.com/products")
        products = response.json().get("products", [])
        print(f"API returned {len(products)} products")
        return products
    except Exception as e:
        print(f"API Error: {e}")
        return []

@tool
def search_pinecone(query_input: dict) -> list:
    """
    Search semantically similar products from Pinecone.
    Do NOT apply external filters here — agent handles filtering internally.
    
    Args:
        query_input: dict with 'description' key containing the search query
    """
    try:
        # Extract the query from the dictionary
        if "description" not in query_input:
            raise ValueError("Missing 'description' key in query input")
            
        query = str(query_input["description"])
        if not query.strip():
            raise ValueError("Query description cannot be empty.")

        print(f"Pinecone searching for: '{query}'")

        # Embed query
        model = SentenceTransformer("all-MiniLM-L6-v2")
        query_emb = model.encode(query).tolist()

        # Initialize Pinecone index
        pc = Pinecone(api_key=config("PC_API_KEY"))
        index = pc.Index("ecommerce-products")

        # Query Pinecone without external filters
        results = index.query(
            vector=query_emb,
            top_k=50,
            include_metadata=True
        )

        products = [match["metadata"] for match in results["matches"]]
        print(f"Pinecone found {len(products)} matches")
        return products

    except Exception as e:
        print(f"Pinecone Error: {e}")
        return []