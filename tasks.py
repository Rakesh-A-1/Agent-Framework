from crewai import Task
from agents import knowledge_agent,retrieval_agent,verification_agent

task_decision = Task(
    description=(
        "Analyze this user query: '{query}' and decide the most appropriate data source. "
        "Return ONLY one word: 'API', 'Pinecone', or 'Hybrid'. "
        "Rules for decision:\n"
        "- If the query is generic (e.g., 'get all products', 'list all items', 'show everything'), ALWAYS choose API.\n"
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
        "Retrieve products based on the source returned by the decision task ('API', 'Pinecone', or 'Hybrid'). "
        "Rules for retrieval:\n"
        "- If 'API', call only fetch_from_api, applying numeric/logical filters.\n"
        "- If 'Pinecone', call only search_pinecone with the ORIGINAL user query '{query}' - do NOT change it to 'beauty products'.\n"
        "- Only include numeric/logical filters in Pinecone if they are explicitly mentioned in the user query.\n"
        "- If 'Hybrid', call BOTH fetch_from_api and search_pinecone (with original query), merge results, remove duplicates."
    ),
    agent=retrieval_agent,
    expected_output="A list of product objects matching the query and filters."
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
