from crewai import Agent,LLM
from decouple import config
from tools import fetch_from_api, search_pinecone

openai_llm = LLM(
    model="gpt-4o-mini"
)
# Knowledge Agent – Decides whether to use API or Pinecone
knowledge_agent = Agent(
    role="Knowledge Agent",
    goal="Decide whether to use API, Pinecone, or BOTH for hybrid search.",
    backstory=(
        "You decide which tool to use based on the user's query. "
        "Use API for exact matches or queries involving numeric/logical filters "
        "(price, stock, rating). "
        "Use Pinecone for semantic meaning searches. "
        "Use Hybrid when query combines semantic meaning + numeric/logical filters."
    ),
    verbose=True,
    llm=openai_llm,
    tools=[fetch_from_api, search_pinecone]
)

retrieval_agent = Agent(
    role="Retrieval Agent",
    goal="Retrieve products from the chosen data source (API, Pinecone, or Hybrid) and ensure relevance automatically.",
    backstory=(
        "Call the tool(s) according to the decision from Knowledge Agent. "
        "If source is 'API', call fetch_from_api with the user query as description. "
        "If source is 'Pinecone', call search_pinecone with the ORIGINAL user query text. "
        "If source is 'Hybrid', call BOTH fetch_from_api and search_pinecone with the original query, "
        "merge results, remove duplicates, and automatically select only relevant products based on the query's intent, "
        "including any numeric/logical conditions (e.g., stock < 10) or semantic keywords (e.g., 'snacking'). "
        "Do NOT apply filters manually in the tool functions; handle all relevance logic within the agent itself."
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
        "You ensure accuracy and relevance by cross-checking API results against Pinecone or vice versa. "
        "For all queries except generic 'all products', automatically filter products based on numeric/logical conditions "
        "(like stock, price, rating) and semantic relevance (keywords in title, description, or tags). "
        "Always return a clean, valid JSON array with exact fields: title, brand, category, price, rating, thumbnail. "
        "No additional text or formatting; agents must handle all filtering automatically."
    ),
    verbose=True
)