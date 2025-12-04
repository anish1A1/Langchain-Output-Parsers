from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


llm = ChatOllama(
    model='llama3.1',
    temperature=0
)

parser = JsonOutputParser()

template = PromptTemplate(
    template='Give me the name, age and city of a fictional person \n {format_instruction}',
    input_variables=[],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# We can use this,
# prompt = template.format()
# result = llm.invoke(prompt)
# # print(result)
# final_result = parser.parse(result.content)


# We can also use it by chaining each other.
chain = template | llm | parser
result = chain.invoke({})
# we are sending blank dict. because it needs input variable value of PromptTemplate

print(result)
print('\n', result['name'])
print(type(result))