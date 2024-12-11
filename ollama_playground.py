from ollama_playground import chat, Tool
from ollama_playground import ChatResponse

# response: ChatResponse = chat(model='llama3.2:3b', messages=[
#   {
#     'role': 'user',
#     'content': 'Why is the sky blue?',
#   },
# ])
# print(response['message']['content'])
# # or access fields directly from the response object
# print(response.message.content)


def addition_func(param1: int, param2: int) -> str:
    return f"The answer is, of course, {param1 + param2}!"


response: ChatResponse = chat(
    model="llama3.2:3b",
    messages=[
        {
            "role": "user",
            "content": "what is 1 plus 1",
        },
    ],
    tools=[Tool(
       function=Tool.Function(
          name='addition_func', 
          description='Add two numbers',
          parameters=Tool.Function.Parameters(
             required=["param1", "param2"],
             properties={
                "param1": Tool.Function.Parameters.Property(
                   type="int",
                   description="The first number to add",
                ),
                "param2": Tool.Function.Parameters.Property(
                   type="int",
                   description="The second number to add",
                ),
             }
          )),

    )],
)

available_functions = {
  'addition_func': addition_func,
}

# for tool in response.message.tool_calls or []:
#   function_to_call = available_functions.get(tool.function.name)
#   if function_to_call:
#     print('Function output:', function_to_call(**tool.function.arguments))
#   else:
#     print('Function not found:', tool.function.name)

print(response['message']['content'])
# or access fields directly from the response object
print(response.message.content)