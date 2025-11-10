from crewai import Task
from agents import knowledge_agent,retrieval_agent,verification_agent

task_decision = Task(
    description="Analyze the user query and decide which data source to use — factual (API) or semantic (Pinecone). "
                "Return ONLY one word: 'API' or 'Pinecone'.",
    agent=knowledge_agent,
    expected_output="Either 'API' or 'Pinecone'."
)


task_retrieval = Task(
    description="Use the chosen source to retrieve matching products for the query.",
    agent=retrieval_agent,
    expected_output="A list of matching product objects."
)

task_verification = Task(
    description=(
        "Verify and clean the retrieved product list. "
        "If the query is generic like 'all products', 'show me all products', or 'give all product', "
        "do not filter; return all retrieved products. "
        "Otherwise, only include products that match the user intent or category from the query."
    ),
    agent=verification_agent,
    expected_output="A list of verified product dictionaries relevant to the query."
)
