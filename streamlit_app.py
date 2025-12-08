import streamlit as st
import json
from crewai import Crew
from tasks import task_decision, task_retrieval, task_verification
from agents import knowledge_agent, retrieval_agent, verification_agent
from product_schema import ProductSchema
from decouple import config
from crewai.memory.external.external_memory import ExternalMemory
from custom_storage import CustomStorage
import pprint
external_memory = ExternalMemory(storage=CustomStorage())

# Initialize the crew
crew = Crew(
    agents=[knowledge_agent, retrieval_agent, verification_agent],
    tasks=[task_decision, task_retrieval, task_verification],
    verbose=True,
    # memory=True,
    external_memory=external_memory,
    embedder={
        "provider": "google-generativeai",
        "config": {
            "api_key": "AIzaSyANDYfX_z46KrGM6dB_HnbwiDXM46dyXQ4",
            "model_name": "gemini-embedding-001"
        }
    },
    process="sequential",
    output_log_file=True,
    tracing=True
)

def search_products(user_query: str):
    """Execute the full product search pipeline with validation."""
    try:
        result = crew.kickoff(inputs={'query': user_query})

        # Extract raw data from Crew output
        raw_data = result.raw if hasattr(result, "raw") else result

        # Convert string to Python list if needed
        if isinstance(raw_data, str):
            raw_data = json.loads(raw_data)

        # Validate each product using Pydantic ProductSchema
        products = [ProductSchema.model_validate(item).model_dump() for item in raw_data]

        return products

    except Exception as e:
        st.error(f"Crew execution error: {e}")
        return []

st.set_page_config(page_title="CrewAI E-Commerce Search", layout="wide")

st.title("🛍️ CrewAI E-Commerce Search Assistant")
st.markdown(
    "Enter a product query — the agents will decide whether to use the API or Pinecone, "
    "retrieve results, verify them, and display the best-matched items."
)

# Input
query = st.text_input("🔍 What are you looking for?", placeholder="e.g., iPhone, laptops, similar to Samsung, all products")

# Run Search
if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a query before searching.")
    else:
        with st.spinner("🤖 Agents are collaborating..."):
            try:
                products = search_products(query)
                st.subheader("🧠 External Memory Debug")
                pprint.pprint(external_memory.storage.memories)
                # You can also just show it in Streamlit:
                st.json(external_memory.storage.memories)
                if products and len(products) > 0:
                    st.success(f"✅ Found {len(products)} products for query: '{query}'")
                    for p in products:
                        with st.container():
                            cols = st.columns([1, 3])
                            with cols[0]:
                                st.image(p.get("thumbnail", ""), width='stretch')
                            with cols[1]:
                                st.subheader(p.get("title", "Unnamed Product"))
                                st.markdown(f"**Brand:** {p.get('brand', 'Unknown')}")
                                st.markdown(f"**Category:** {p.get('category', '-')}")
                                st.markdown(f"**Price:** ${p.get('price', 0)}")
                                st.markdown(f"⭐ Rating: {p.get('rating', 0)} / 5")
                            st.divider()
                else:
                    st.warning("No products found.")
            except Exception as e:
                st.error(f"⚠️ Error during search: {e}")

# Footer
st.markdown("---")
st.caption("Powered by CrewAI Agents + Pinecone + DummyJSON API")