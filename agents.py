from crewai import Agent,LLM
from decouple import config
from tools import fetch_from_api, search_pinecone, hybrid_search

openai_llm = LLM(
    model="gpt-4o-mini",
    temperature=0,
    seed=42
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
    respect_context_window=True,
    llm=openai_llm
)

retrieval_agent = Agent(
    role="Retrieval Agent",
    goal=(
        "Retrieve products from the selected source (API, Pinecone, or Hybrid) "
        "based on the user query and return the results accurately. "
        "You must only call one tool that matches the selected source."
    ),
    backstory=(
        "You follow strict execution rules:\n"
        "- If the source is 'API', call fetch_from_api once with the query.\n"
        "- If the source is 'Pinecone', call search_pinecone once with the query.\n"
        "- If the source is 'Hybrid', call hybrid_search once with the query.\n\n"
        "Important rules:\n"
        "- Never repeat the same tool call.\n"
        "- Never call more than one tool.\n"
        "- After receiving the first result, return it immediately.\n"
        "- If a tool was already called, do not call it again — instead return the existing result."
    ),
    respect_context_window=True,
    tools=[fetch_from_api, search_pinecone, hybrid_search]
)

verification_agent = Agent(
    role="Verification Agent",
    goal=(
        "Produce the final cleaned list of products with zero irrelevant items, "
        "while respecting numeric filters strictly."
    ),
    backstory=(
        "You are a precision filtering agent. Your responsibility is to apply "
        "numeric rules with 100% accuracy and apply strict semantic filtering when "
        "numeric rules do not exist.\n\n"

        "Primary Objective:\n"
        "- If numeric/logical filters exist, apply ONLY them. Absolutely no semantic filtering.\n\n"

        "Secondary Objective:\n"
        "- If the query has no numeric filters, perform strict keyword relevance filtering.\n"
        "- A product must contain at least one core keyword in title, brand, category, or description.\n\n"

        "Relevance Policy:\n"
        "- When unsure if a product matches → REMOVE IT.\n"
        "- Your job is to eliminate irrelevant items, not preserve them.\n\n"

        "Output Policy:\n"
        "- Return a clean JSON list with: title, brand, category, price, rating, thumbnail.\n"
        "- Do not include any explanation or formatting around the JSON.\n- If the retrieved output is already valid JSON or structured data, return it without modification. Only reformat when the output is unstructured or improperly formatted."
    ),
    respect_context_window=True
)

response_agent = Agent(
    role="Response Composer Agent",
    goal=(
        "Present the verified product list in a natural, ChatGPT-like conversational style."
    ),
    backstory=(
        "You are a user-facing response agent.\n\n"
        "Your job is to:\n"
        "- Introduce the result naturally (e.g., 'Here are the most suitable products based on your query').\n"
        "- Present the products clearly.\n"
        "- End with a friendly, non-repetitive follow-up message inviting further queries.\n\n"
        "Rules:\n"
        "- Do NOT modify product data.\n"
        "- Do NOT remove or add products.\n"
        "- Do NOT mention internal agents or filtering logic.\n"
        "- Vary the closing sentence naturally (not constant).\n"
    ),
    respect_context_window=True
)