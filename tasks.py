from crewai import Task
from agents import knowledge_agent, retrieval_agent, verification_agent, response_agent

task_decision = Task(
    description=(
        "Analyze the user query: '{query}' and the conversation history below:\n"
        "HISTORY:\n{chat_history}\n\n"
        "YOUR JOB:\n"
        "1. If the query is a follow-up (e.g., 'give more', 'cheaper', 'what about Samsung?'), "
        "   use the HISTORY to construct a specific, standalone query.\n"
        "   - Example: 'Cheaper' -> 'Fragrances under $50'\n"
        "   - Example: 'Give more' -> 'Fragrances between $50 and $100' (keep consistent)\n"
        "2. If it is a new topic, use the query as is.\n\n"
        "OUTPUT FORMAT:\n"
        "Return a JSON object: {{'source': 'API'|'Pinecone'|'Hybrid', 'refined_query': 'YOUR_CALCULATED_QUERY_STRING', 'reason': '...'}}"
    ),
    agent=knowledge_agent,
    expected_output="JSON object with 'source' and 'refined_query'"
)

task_retrieval = Task(
    description=(
        "You are the execution arm. Do NOT look at the user input directly.\n"
        "1. Read the output from the 'Knowledge Agent' (provided in context).\n"
        "2. Extract the 'source' and 'refined_query'.\n"
        "3. Call the appropriate tool using ONLY the 'refined_query'.\n\n"
        "Execution Rules:\n"
        "- API: call fetch_from_api(refined_query)\n"
        "- Pinecone: call search_pinecone(refined_query)\n"
        "- Hybrid: call hybrid_search(refined_query)\n"
    ),
    agent=retrieval_agent,
    context=[task_decision], 
    expected_output="List of products from the tool"
)

task_verification = Task(
    description=(
        "You are the final validator. Validate the retrieved products against the 'refined_query'.\n"
        "You also have access to the conversation history below:\n"
        "----------------\n"
        "{chat_history}\n" 
        "----------------\n\n"

        "1. Get 'refined_query' from Knowledge Agent context.\n"
        "2. Get product list from Retrieval Agent context.\n\n"

        "STRICT FILTERING RULES:\n"
        "   - Apply numeric/brand filters from 'refined_query' (e.g., price < 50).\n"
        "   - **DEDUPLICATION RULE**: Check the 'chat_history'. If a product's Title is already mentioned in the Assistant's previous messages, REMOVE IT. We must show NEW products only.\n"
        "   - If no products remain after filtering/deduplication, return an empty list [].\n\n"

        "4. Return ONLY the valid JSON list."
    ),
    agent=verification_agent,
    context=[task_decision, task_retrieval], 
    expected_output="JSON list of valid products"
)

task_response = Task(
    description=(
        "Write a natural response to the user based on the verified product list.\n"
        "Context: User asked '{query}'. We found these products: [See Context].\n"
        "If the list is empty, apologize and suggest alternatives."
    ),
    agent=response_agent,
    context=[task_verification],
    expected_output="Conversational response"
)