from crewai import Agent,LLM
from decouple import config
from tools import fetch_from_api, search_pinecone

openai_llm = LLM(
    model="gpt-4o-mini"
)
# Knowledge Agent – Decides whether to use API or Pinecone
knowledge_agent = Agent(
    role="Knowledge Agent",
    goal="Decide whether to fetch data from API or Pinecone.",
    backstory=(
        "Use API for: empty queries, generic product requests ('all products', 'get all products', 'show me everything'), "
        "exact product names (like 'iPhone 15'), exact brand names (like 'Samsung'), exact category names (like 'electronics'). "
        "Use Pinecone for ALL semantic searches: descriptive terms (like 'beauty items', 'affordable phones'), "
        "comparisons (like 'price < 50'), similarity searches (like 'similar to iPhone'), "
        "and any queries with adjectives or descriptive language."
    ),
    verbose=True,
    llm=openai_llm,
    tools=[fetch_from_api, search_pinecone]
)

retrieval_agent = Agent(
    role="Retrieval Agent",
    goal="Retrieve data from the chosen source using the EXACT query string.",
    backstory=(
        "You are expert at using tools correctly. "
        "ALWAYS pass the query as a simple string value. "
        "NEVER use complex objects with 'description' or 'type' fields. "
        "Just pass the query string directly to the tool."
    ),
    verbose=True,
    llm=openai_llm,
    tools=[fetch_from_api, search_pinecone]
)


# Verification Agent – Cross-check and finalize output
verification_agent = Agent(
    role="Verification Agent",
    goal="Validate, refine, and finalize the product results before returning to the user.",
    backstory=(
        "You ensure accuracy by checking API results against Pinecone or vice versa "
        "to confirm up-to-date details and consistency."
    ),
    verbose=True
)