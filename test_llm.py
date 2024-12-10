from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
import os

llm = ChatGroq(
    temperature=0,
    model_name="llama-3.1-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# Pass a list of messages, at least one HumanMessage
messages = [HumanMessage(content="Hello, how are you?")]
response = llm(messages)
print("LLM Response:", response)
