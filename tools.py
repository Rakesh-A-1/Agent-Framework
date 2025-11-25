from crewai.tools import tool
import requests
from decouple import config
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import streamlit as st

@st.cache_resource
def load_st_model():
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    model.encode(["warmup"])
    return model

@st.cache_resource
def load_pinecone():
    return Pinecone(api_key=config("PC_API_KEY"))

# Internal helper functions (not tools)
def _fetch_from_api_helper(query: str) -> list:
    """Helper function to fetch from API"""
    try:
        print(f"API searching for: {query}")
        
        # Get all products first
        url = "https://dummyjson.com/products"
        response = requests.get(url)
        all_products = response.json().get("products", [])
            
        print(f"API returned {len(all_products)} products")
        return all_products
    except Exception as e:
        print(f"API Error: {e}")
        return []

def _search_pinecone_helper(query: str) -> list:
    """Helper function to search Pinecone"""
    try:
        if not query.strip():
            raise ValueError("Query description cannot be empty.")

        print(f"Pinecone searching for: '{query}'")

        # Embed query
        model = load_st_model()
        query_emb = model.encode(query).tolist()

        # Initialize Pinecone index
        pc = load_pinecone()
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

# Tool functions
@tool
def fetch_from_api(query: str) -> list:
    """Fetch products from DummyJSON API"""
    return _fetch_from_api_helper(query)

@tool
def search_pinecone(query: str) -> list:
    """Search semantically similar products from Pinecone"""
    return _search_pinecone_helper(query)

@tool
def hybrid_search(query: str) -> list:
    """Perform hybrid search by combining API fetch and semantic search"""
    try:
        print(f"Hybrid searching for: '{query}'")
        
        # Get results from both sources using helper functions
        api_results = _fetch_from_api_helper(query)
        semantic_results = _search_pinecone_helper(query)
        
        # Combine results
        all_results = api_results + semantic_results
        
        # Basic deduplication by product ID
        seen_ids = set()
        unique_results = []
        
        for product in all_results:
            product_id = product.get('id')
            if product_id and product_id not in seen_ids:
                seen_ids.add(product_id)
                unique_results.append(product)
            elif not product_id:
                unique_results.append(product)
        
        print(f"Hybrid search found {len(unique_results)} unique products")
        return unique_results
        
    except Exception as e:
        print(f"Hybrid search error: {e}")
        return []