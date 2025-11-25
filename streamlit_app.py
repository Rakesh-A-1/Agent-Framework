import streamlit as st
import json
from crewai import Crew
from tasks import task_decision, task_retrieval, task_verification
from agents import knowledge_agent, retrieval_agent, verification_agent

# Initialize the crew
crew = Crew(
    agents=[knowledge_agent, retrieval_agent, verification_agent],
    tasks=[task_decision, task_retrieval, task_verification],
    verbose=True,
    process="sequential"
)

def search_products(user_query: str):
    """Execute the full product search pipeline"""
    try:
        result = crew.kickoff(inputs={'query': user_query})
        return result
    except Exception as e:
        st.error(f"Crew execution error: {e}")
        return None

def parse_final_output(output):
    """Convert Crew/LLM output into a valid Python list."""
    
    # If Crew result object, unwrap to raw text
    if hasattr(output, "raw"):
        output = output.raw
    
    # If it's already a list, return as-is
    if isinstance(output, list):
        return output

    # If it's a string, attempt JSON parsing
    if isinstance(output, str):
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            # fallback: try removing whitespace or stray characters
            cleaned = output.strip()
            try:
                return json.loads(cleaned)
            except:
                return []

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
                result = search_products(query)
                if result:
                    # Get the final output from the last task
                    final_output = result.raw if hasattr(result, 'raw') else result
                    products = parse_final_output(final_output)
                    if isinstance(products, list) and len(products) > 0:
                        st.success(f"✅ Found {len(products)} products for query: '{query}'")
                    for p in products:
                        with st.container():
                            cols = st.columns([1, 3])
                            with cols[0]:
                                st.image(p.get("thumbnail", ""), use_container_width=True)
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