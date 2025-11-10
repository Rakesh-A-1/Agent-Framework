from pinecone import Pinecone
from decouple import config
from sentence_transformers import SentenceTransformer
import requests

pc = Pinecone(api_key=config("PC_API_KEY"))
index_name = "ecommerce-products"
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create the index if it doesn’t exist
if index_name not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,          # same as MiniLM embedding dimension
        metric="cosine",
        spec={
            "serverless": {
                "cloud": "aws",
                "region": "us-east-1"
            }
        }
    )
    print(f"Created Pinecone index: {index_name}")
else:
    print(f"Index '{index_name}' already exists")

# Add products from DummyJSON to the index
def add_products():
    response = requests.get("https://dummyjson.com/products")
    data = response.json()
    products = data.get("products", [])
    index = pc.Index(index_name)

    for p in products:
        text = f"""
        {p.get('title', '')}.
        {p.get('description', '')}.
        Category: {p.get('category', '')}.
        Brand: {p.get('brand', 'Unknown')}.
        """
        emb = model.encode(text).tolist()
        index.upsert([
            (
                str(p["id"]),
                emb,
                {
                    "title": p.get("title", ""),
                    "category": p.get("category", ""),
                    "brand": p.get("brand", "Unknown"),
                    "price": p.get("price", 0),
                    "rating": p.get("rating", 0),
                    "thumbnail": p.get("thumbnail", "")
                }
            )
        ])

    print(f"Indexed {len(products)} products to Pinecone")

if __name__ == "__main__":
    add_products()