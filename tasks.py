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
        "Verify and clean the retrieved product list for query: '{query}'. "
        "If the query is generic like 'all products', 'show me all products', or 'give all product', "
        "do not filter; return all retrieved products. "
        "Otherwise, only include products that match the user intent or category from the query."
        "\n\n**COMPULSORY OUTPUT FORMAT:**"
        "\nReturn ONLY a valid JSON array of product objects with these exact fields:"
        "\n- title (string)"
        "\n- brand (string)" 
        "\n- category (string)"
        "\n- price (number)"
        "\n- rating (number)"
        "\n- thumbnail (string)"
        "\n\n**STRICTLY FOLLOW:**"
        "\n- No additional text before or after the JSON array"
        "\n- No markdown formatting"
        "\n- Valid JSON only"
        "\n- Include ALL required fields for each product"
    ),
    agent=verification_agent,
    expected_output="A valid JSON array of product objects with exact fields: title, brand, category, price, rating, thumbnail"
)
