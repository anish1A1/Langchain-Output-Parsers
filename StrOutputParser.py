from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id= "moonshotai/Kimi-K2-Thinking",
#     task="text-generation"
# )
# This model has structured Output fuction and the model below doesn't has structured Output fuction

llm = HuggingFaceEndpoint(
    repo_id= "google/gemma-2-2b-it",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)


# We are going to build a apllication where we get a topic, then LLM will convert it into detailed report, then LLM will again give 5 lines summary of it.

# 1st promt --> Detailed Report

template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)


#  2nd Prompt --> Summary
template2 = PromptTemplate(
    template='Write a 5 line summary on the following text. /n {text}',
    input_variables=['text']
)

prompt1 = template1.invoke({'topic':'black hole'})
result = model.invoke(prompt1)

prompt2 = template2.invoke({'text': result.content})
result2 = model.invoke(prompt2)

print(result2.content)