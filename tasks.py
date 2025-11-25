from crewai import Task
from agents import knowledge_agent,retrieval_agent,verification_agent

task_decision = Task(
    description=(
        "Analyze this user query: '{query}' and decide the most appropriate data source. "
        "Return ONLY one word: 'API', 'Pinecone', or 'Hybrid'.\n\n"
        "Rules for decision:\n"
        "- If the query is one of the following generic queries, ALWAYS choose API:\n"
        "  'get all products', 'list all items', 'show everything', 'all products', "
        "'show me all products', 'give all product'\n"
        "- If the query is an exact match for a product title, brand, or category, choose API.\n"
        "- If the query contains numeric/logical filters (price, stock, rating), choose API.\n"
        "- If the query is semantic/meaning-based (e.g., 'beauty items', 'long-lasting phones', 'premium skincare'), choose Pinecone.\n"
        "- If the query combines semantic meaning AND numeric/logical filters, choose Hybrid."
    ),
    agent=knowledge_agent,
    expected_output="One word: 'API', 'Pinecone', or 'Hybrid'."
)

task_retrieval = Task(
    description=(
        "Retrieve products based on the source returned by the decision task ('API', 'Pinecone', or 'Hybrid').\n\n"
        "Rules for retrieval:\n"
        "- If 'API', fetch all products from the API. Do NOT manually filter here; the Verification Agent will handle any numeric/logical or category-based filtering.\n"
        "- If 'Pinecone', call search_pinecone with the ORIGINAL user query '{query}'. Do NOT modify the query or apply filters outside of agent logic.\n"
        "- If 'Hybrid', call BOTH fetch_from_api and search_pinecone with original query, merge results, remove duplicates, but do NOT manually filter in code.\n\n"
        "Special Case - Generic Queries:\n"
        "  If the query matches: 'get all products', 'list all items', 'show everything', 'all products', "
        "'show me all products', 'give all product', treat it as an API query returning all products without filtering.\n\n"
        "All filtering, validation, and selection of relevant products must be performed automatically by the agents. "
        "Do not implement any manual filter logic in the tool functions."
    ),
    agent=retrieval_agent,
    expected_output="A list of product objects matching the query, with filtering handled internally by the agents."
)

task_verification = Task(
    description=(
        "You are the final validator. Use the user's query: '{query}' and the "
        "retrieved list of products.\n\n"

        "STRICT RULES:\n\n"

        "1. Extract core keywords from the query (nouns + adjectives only). "
        "Examples:\n"
        "- 'matte lipstick' → ['matte', 'lipstick']\n"
        "- 'gaming laptop under 700' → ['gaming', 'laptop']\n"
        "- 'snacks for kids' → ['snacks', 'kids']\n\n"

        "2. Detect numeric/logical conditions, such as:\n"
        "- price < X, price <= X, price > X, price >= X\n"
        "- rating < X, rating > X\n"
        "- stock < X, stock <= X\n\n"

        "3. If ANY numeric/logical condition exists:\n"
        "- APPLY ONLY the numeric rules.\n"
        "- DO NOT apply semantic filtering.\n"
        "- DO NOT remove any product that satisfies the numeric rule.\n\n"

        "4. If NO numeric/logical conditions exist:\n"
        "- A product MUST contain at least one core keyword in any of these fields:\n"
        "  title, description, category, brand.\n"
        "- Case-insensitive matching.\n"
        "- If none of the keywords match → REMOVE the product.\n"
        "- This prevents irrelevant results.\n\n"

        "5. Fail-safe rule (updated):\n"
        "- When unsure, REMOVE the product.\n"
        "- Do NOT keep borderline items.\n"
        "- Only keep items that clearly match the query.\n\n"

        "6. You must NOT hallucinate new products. Only use the retrieved list.\n\n"

        "7. Return ONLY a clean JSON array containing objects with fields:\n"
        "   title, brand, category, price, rating, thumbnail\n"
        "No markdown. No explanations. No additional text."
    ),
    agent=verification_agent,
    expected_output="A valid JSON array of product objects"
)