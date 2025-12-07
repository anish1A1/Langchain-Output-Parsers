from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field


# We are creating a Joke about a particular topic. 
# The joke should have setup of the joke and punchline


llm = HuggingFaceEndpoint(
    repo_id= "moonshotai/Kimi-K2-Thinking",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)



class Joke(BaseModel):
    setup:str = Field("the setup of the joke")
    punchline:str = Field("the punchline of the joke")
    

parser = JsonOutputParser(pydantic_object=Joke)

template = PromptTemplate(
    template='Tell me a sarcastic joke about {topic} /n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# prompt = template.invoke({'topic':'pokemon'})
# print(prompt)

chain = template | model | parser

output = chain.invoke({'topic':'pokemon'})

print(output)