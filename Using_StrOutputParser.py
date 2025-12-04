from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = HuggingFaceEndpoint(
    repo_id= "google/gemma-2-2b-it",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

template2 = PromptTemplate(
    template='Write a 5 line summary on the following text /n {text}',
    input_variables=['text']
)
parser = StrOutputParser()
# The StrOutputParser will parse the response of the llm models into string (It's just like result.content from a model.)

chain = template1 | model | parser | template2 | model | parser

# To form a chain, we first invoke template1 and provide it to the model (here model is also invoked now), then we invoke the parser provideing the vlaues from model to parser. Here the parser will provide String output to the template2 and it keeps repeating.
 
result = chain.invoke({
    'topic' : 'black hole'
})
print(result)