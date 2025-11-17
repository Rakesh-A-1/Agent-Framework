from crewai.tools import tool
from decouple import config
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import requests

#helper function
def matches_filters(p, filters):
            """Return True if product matches all provided filters."""
            # Numeric filters
            if "price" in filters:
                if "min" in filters["price"] and p.get("price", 0) < filters["price"]["min"]:
                    return False
                if "max" in filters["price"] and p.get("price", 0) > filters["price"]["max"]:
                    return False
            if "rating" in filters:
                if "min" in filters["rating"] and p.get("rating", 0) < filters["rating"]["min"]:
                    return False
                if "max" in filters["rating"] and p.get("rating", 0) > filters["rating"]["max"]:
                    return False
            if "stock" in filters:
                if "min" in filters["stock"] and p.get("stock", 0) < filters["stock"]["min"]:
                    return False
                if "max" in filters["stock"] and p.get("stock", 0) > filters["stock"]["max"]:
                    return False

            # String filters
            if "brand" in filters and p.get("brand", "").lower() != filters["brand"].lower():
                return False
            if "category" in filters and p.get("category", "").lower() != filters["category"].lower():
                return False

            return True

@tool
def fetch_from_api(query: dict) -> list:
    """
        Fetch product data from DummyJSON API with optional filters.
        
        query can include:
        {
            "description": "red lipstick",  # optional, exact match on title/brand/category
            "filters": {                     # optional, filter on metadata
                "price": {"min": 5, "max": 20},
                "rating": {"min": 4},
                "stock": {"min": 10},
                "brand": "Essence",
                "category": "beauty"
            }
        }
    """
    try:
        description = str(query.get("description") or "").lower().strip()
        filters = query.get("filters", {})

        # Fetch all products from API
        response = requests.get("https://dummyjson.com/products")
        products = response.json().get("products", [])

        # Filter products based on description and filters
        result = []
        for p in products:
            title = p.get("title", "").lower()
            brand = p.get("brand", "").lower()
            category = p.get("category", "").lower()

            # Match description if provided, otherwise treat as all
            if description == "" or description in [title, brand, category]:
                if matches_filters(p, filters):
                    result.append(p)

        print(f"API found {len(result)} products matching query + filters")
        return result

    except Exception as e:
        print(f"API Error: {e}")
        return []

@tool
def search_pinecone(query_input: str, filters: dict = {}) -> list:
    """
    Search semantically similar products from Pinecone with optional numeric/logical filters.
    - query_input: a plain string describing the product search.
    - filters: dict, e.g. {"price": {"$lte": 20}, "rating": {"$gte": 4.5}}
    """
    try:
        # Ensure query_input is a string
        if isinstance(query_input, dict):
            # Extract a string from possible keys
            query = query_input.get("description") or query_input.get("query_input") or ""
        else:
            query = str(query_input)

        if not query:
            raise ValueError("Query input cannot be empty.")

        print(f"Pinecone searching for: '{query}' with filters: {filters}")

        # Embed query
        model = SentenceTransformer("all-MiniLM-L6-v2")
        query_emb = model.encode(query).tolist()

        # Initialize Pinecone index
        pc = Pinecone(api_key=config("PC_API_KEY"))
        index = pc.Index("ecommerce-products")

        # Query Pinecone with optional filters
        results = index.query(
            vector=query_emb,
            top_k=50,
            include_metadata=True,
            filter=filters  # apply numeric/logical filters in Pinecone itself
        )

        products = [match["metadata"] for match in results["matches"]]
        print(f"Pinecone found {len(products)} matches")
        return products

    except Exception as e:
        print(f"Pinecone Error: {e}")
        return []