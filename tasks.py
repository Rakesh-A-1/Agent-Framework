from crewai import Task
from agents import knowledge_agent,retrieval_agent,verification_agent

task_decision = Task(
    description="Analyze this user query: '{query}' and decide which data source to use. "
                "Return ONLY one word: 'API' or 'Pinecone'. "
                "API is ONLY for exact matches like 'iPhone 15' or 'Samsung'. "
                "Pinecone is for semantic searches like 'beauty items' or 'affordable phones'.",
    agent=knowledge_agent,
    expected_output="Either 'API' or 'Pinecone'."
)

task_retrieval = Task(
    description="Use the chosen source to retrieve matching products for this exact query: '{query}'.",
    agent=retrieval_agent,
    expected_output="A list of matching product objects."
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
