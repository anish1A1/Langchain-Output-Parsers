# from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema

# The Structured Output parser, responseSchema doesn't work in latest langchain


from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from pydantic import BaseModel, Field

llm = HuggingFaceEndpoint(
    repo_id= "moonshotai/Kimi-K2-Thinking",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)


# class Info()

