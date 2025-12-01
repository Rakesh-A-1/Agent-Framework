from pydantic import BaseModel

class ProductSchema(BaseModel):
    title: str
    brand: str
    category: str
    price: float
    rating: float
    thumbnail: str