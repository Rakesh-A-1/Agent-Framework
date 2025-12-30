import streamlit as st
import json
from crewai import Crew
from tasks import task_decision, task_retrieval, task_verification,task_response
from agents import knowledge_agent, retrieval_agent, verification_agent,response_agent
from product_schema import ProductSchema
from crewai.memory.external.external_memory import ExternalMemory
from custom_storage import FileStorage
from crewai import LLM
from crewai.utilities.prompts import Prompts

openai_llm = LLM(
    model="gpt-4o-mini",
    temperature=0.7   # allow natural language variation
)

agents = [knowledge_agent, retrieval_agent, verification_agent,response_agent]
tasks = [task_decision, task_retrieval, task_verification,task_response]

# Create the prompt generator
for agent, task in zip(agents, tasks):
    prompt_generator = Prompts(
        agent=agent,
        has_tools=len(agent.tools) > 0,
        use_system_prompt=agent.use_system_prompt
    )

    # Generate and inspect the actual prompt
    generated_prompt = prompt_generator.task_execution()

    # Print the complete system prompt that will be sent to the LLM
    if "system" in generated_prompt:
        print("=== SYSTEM PROMPT ===")
        print(generated_prompt["system"])
        print("\n=== USER PROMPT ===")
        print(generated_prompt["user"])
    else:
        print("=== COMPLETE PROMPT ===")
        print(generated_prompt["prompt"])

    # You can also see how the task description gets formatted
    print("\n=== TASK CONTEXT ===")
    print(f"Task Description: {task.description}")
    print(f"Expected Output: {task.expected_output}")

external_memory = ExternalMemory(storage=FileStorage())

# Initialize the crew
crew = Crew(
    agents=[knowledge_agent, retrieval_agent, verification_agent,response_agent],
    tasks=[task_decision, task_retrieval, task_verification,task_response],
    verbose=True,
    external_memory=external_memory,
    process="sequential",
    output_log_file=True,
    tracing=True
)

# Page config
st.set_page_config(
    page_title="E-commerce AI Agent",
    page_icon="🛍️",
    layout="wide"
)

# Title and Description
st.title("🛍️ Crew E-commerce AI Agent")
st.markdown("""
This agent can help you find products using:
- **Structured Data**: Prices, stock, IDs (via DummyJSON)
- **Semantic Search**: Descriptions, features (via Pinecone Vector DB)
- **Hybrid Search**: Combining both for complex queries
""")

with st.sidebar: 
    st.header("Configuration") 
    st.info("Ensure your .env file contains required API keys.") 
    if st.button("Clear Chat"): 
        st.session_state.messages = [] 
        st.success("Chat cleared")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask about a product..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history_str = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[-5:]]
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                inputs = {
                    'query': prompt,
                    'chat_history': history_str
                }

                result = crew.kickoff(inputs=inputs)
                
                final_response = result.raw if hasattr(result, 'raw') else str(result)

                st.markdown(final_response)
                st.session_state.messages.append({"role": "assistant", "content": final_response})
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")