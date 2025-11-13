from crewai.tools import tool
from decouple import config
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

@tool
def fetch_from_api(query_input: str) -> list:
    """Fetch product data from the DummyJSON API for exact matches and generic queries."""
    try:
        # Handle different input formats
        if isinstance(query_input, dict):
            query = query_input.get('description', '') or query_input.get('query_input', '')
        else:
            query = str(query_input)
            
        print(f"API searching for: '{query}'")
        
        response = requests.get("https://dummyjson.com/products")
        products = response.json().get("products", [])
        
        # Enhanced generic queries detection
        generic_terms = [
            "all products", "show me all products", "give all products", "everything", "",
            "get all products", "list all products", "all items", "show everything",
            "all", "products", "items", "list products"
        ]
        
        query_lower = query.lower().strip()
        
        # Check if query is generic
        if is_generic_query(query_lower, generic_terms):
            print(f"API returning all {len(products)} products (generic query)")
            return products

        # Exact matches only
        exact_matches = []
        for product in products:
            if (query_lower == product.get('title', '').lower() or
                query_lower == product.get('brand', '').lower() or
                query_lower == product.get('category', '').lower()):
                exact_matches.append(product)
        
        print(f"API found {len(exact_matches)} exact matches")
        return exact_matches
        
    except Exception as e:
        print(f"API Error: {e}")
        return []

@tool
def search_pinecone(query_input: str) -> list:
    """Search semantically similar products from Pinecone."""
    try:
        # Handle different input formats
        if isinstance(query_input, dict):
            query = query_input.get('description', '') or query_input.get('query_input', '')
        else:
            query = str(query_input)
            
        print(f"Pinecone searching for: '{query}'")
        
        model = SentenceTransformer("all-MiniLM-L6-v2")
        pc = Pinecone(api_key=config("PC_API_KEY"))
        index = pc.Index("ecommerce-products")
        query_emb = model.encode(query).tolist()
        results = index.query(vector=query_emb, top_k=20, include_metadata=True)
        
        products = [m["metadata"] for m in results["matches"]]
        print(f"Pinecone found {len(products)} semantic matches")
        return products
        
    except Exception as e:
        print(f"Pinecone Error: {e}")
        return []

def is_generic_query(query: str, generic_terms: list) -> bool:
    """Check if the query is a generic request for all products."""
    # Direct match
    if query in generic_terms:
        return True
    
    # Contains generic keywords
    generic_keywords = ['all', 'every', 'list', 'show', 'get', 'give']
    product_keywords = ['product', 'item', 'things', 'stuff']
    
    has_generic = any(keyword in query for keyword in generic_keywords)
    has_product = any(keyword in query for keyword in product_keywords)
    
    return has_generic and has_product
