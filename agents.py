from crewai import Agent,LLM
from decouple import config
from tools import fetch_from_api, search_pinecone

openai_api_key = config("OPENAI_API_KEY")
openai_llm = LLM(
    model="gpt-4o-mini",  
    api_key=openai_api_key
)

# Knowledge Agent – Decides whether to use API or Pinecone
knowledge_agent = Agent(
    role="Knowledge Agent",
    goal="Decide whether to fetch data from API or Pinecone.",
    backstory=(
        "Understands user queries. "
        "Use API for empty or generic queries (like 'all products', 'show me all products') "
        "or for factual/numeric filters (price, stock, rating). "
        "Use Pinecone for semantic similarity or 'similar to' type queries."
    ),
    verbose=True,
    llm=openai_llm,
    tools=[fetch_from_api, search_pinecone]
)


# Retrieval Agent – Actually fetches the data
retrieval_agent = Agent(
    role="Retrieval Agent",
    goal="Retrieve data from the chosen source.",
    backstory=(
        "Can fetch data using available tools. "
        "If the selected source (API or Pinecone) returns an empty result, "
        "automatically retry using the alternate source and merge both results if relevant."
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