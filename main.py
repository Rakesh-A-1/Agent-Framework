from fastapi import FastAPI
from crewai import Crew
from tasks import task_decision, task_retrieval, task_verification
import uvicorn
import json

app = FastAPI()

crew = Crew(
    tasks=[task_decision, task_retrieval, task_verification]
)

# @app.get("/search")
# def search_products(query: str):
#     result = crew.kickoff(inputs={"query": query})
#     return {"query": query, "result": result}

@app.get("/search")
def search_products(query: str):
    result = crew.kickoff(inputs={"query": query})
    
    # extract the final agent output (the verified one)
    final_output = result.tasks_output[-1].raw  # last task’s raw output

    try:
        # convert string JSON → Python list
        products = json.loads(final_output)
    except:
        products = final_output

    return {"query": query, "products": products}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)