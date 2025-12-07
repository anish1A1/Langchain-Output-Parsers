from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Here we are creating a apllication where it will give name, age and city of a person.


llm = HuggingFaceEndpoint(
    repo_id= "moonshotai/Kimi-K2-Thinking",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)


class Person(BaseModel):
    name:str = Field(description='Name of the Person')
    age:int = Field(description='Age of the Person')
    city:str = Field(description='Name of the city person belongs to')

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Generate the name, age and city of a fictional {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)



prompt = template.invoke({'place':'Nepali'})
print(prompt)

result = model.invoke(prompt)
final_result = parser.parse(result.content)
print(final_result)

# Easy and efficient Method
# chain = template | model | parser
# output = chain.invoke(
#     {'place':'Nepali'}
#     )

# print(output)