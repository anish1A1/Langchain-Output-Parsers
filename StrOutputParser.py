from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id= "moonshotai/Kimi-K2-Thinking",
#     task="text-generation"
# )
# This model has structured Output fuction

llm = HuggingFaceEndpoint(
    repo_id= "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)


# We are going to build a apllication where we get a topic, then LLM will convert it into detailed report, then LLM will again give 5 lines summary of it.

# 1st promt --> Detailed Report

#  2nd Prompt --> Summary
