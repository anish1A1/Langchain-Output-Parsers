from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

llm = HuggingFaceEndpoint(
    repo_id= "moonshotai/Kimi-K2-Thinking",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()



template = PromptTemplate(
    template='Give me the name, age and city of a fictional person \n {format_instruction}',
    input_variables=[],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)
# here the parser.get_format_instructions will add --> Return a JSON object    in the template.

prompt = template.format()
# print(prompt)
# The output of this will be:
# Give me the name, age and city of a fictional person 
# Return a JSON object.


result = model.invoke(prompt)
# print(result)

final_result = parser.parse(result.content)
print(final_result)
print('\n', final_result['name'])
print(type(final_result))