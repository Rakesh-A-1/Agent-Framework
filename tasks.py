from crewai import Task
from agents import knowledge_agent,retrieval_agent,verification_agent,response_agent

task_decision = Task(
    description=(
        "Analyze this user query: '{query}' and decide the most appropriate data source. "
        "Return ONLY a JSON object with the format: {{'source': 'API'|'Pinecone'|'Hybrid', 'reason': 'brief explanation'}}\n\n"
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
    expected_output="JSON object: {'source': 'API'|'Pinecone'|'Hybrid', 'reason': 'brief explanation'}"
)

task_retrieval = Task(
    description=(
        "Based on the source decision from the previous agent, retrieve products for user query: '{query}'\n\n"
        "Execution rules:\n"
        "- If source is 'API': call fetch_from_api ONCE with the exact query string\n"
        "- If source is 'Pinecone': call search_pinecone ONCE with the exact query string\n"
        "- If source is 'Hybrid': call hybrid_search ONCE with the exact query string\n\n"
        "- Execute only one tool matching the source and do not repeat the same tool call.\n"
        "- Once results are received, stop and return them immediately."
        "CRITICAL: \n"
        "1. Call the tool ONLY ONCE - do not repeat the same call\n"
        "2. Pass the query as a plain string\n"
        "3. After getting results, return them immediately - do not call the tool again\n"
        "4. If you get results, your job is done - move to the next step"
    ),
    agent=retrieval_agent,
    expected_output="Raw list of product objects from the chosen data source"
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

task_response = Task(
    description=(
        "User query: '{query}'\n\n"
        "You will receive a verified JSON list of products.\n\n"
        "Generate a natural language response like ChatGPT:\n"
        "- Start with a friendly intro.\n"
        "- Present the products clearly.\n"
        "- End with a polite, varied follow-up (e.g., offering help, refinements, or alternatives).\n\n"
        "Do NOT output JSON.\n"
        "Do NOT change product details."
    ),
    agent=response_agent,
    expected_output="A conversational, user-friendly response"
)