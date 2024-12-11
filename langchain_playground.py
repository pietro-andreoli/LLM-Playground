
from langchain_ollama import ChatOllama
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
import os
import json

secrets = json.load(open("secret.json"))

os.environ["LANGCHAIN_TRACING_V2"]= "true"
os.environ["LANGCHAIN_ENDPOINT"]= "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"]= secrets.get("langsmith_api_key")
os.environ["LANGCHAIN_PROJECT"]= "langchain_playground"

def print_w_role(role: str, content: str):
    print(f"[{role}]: {content}")

users = {
    "Peter Andreoli": {
        "age": 28,
        "address": "123 four street",
        "email": "test@test.com"
    },
    "Teodora Tockovska": {
        "age": 27,
        "address": "905 5th street",
        "email": "tea.toc@gmail.com"
    },
    "Opal Andreoli": {
        "age": 1,
        "address": "1 Dachshund Lane",
        "email": "opal@outlook.com"
    },
}

@tool
def get_user_age(name: str):
    """Gets the age of a user.

    Args:
        name (str): The name of the user to search for.

    Returns:
        int: The age of the user, or None if not found.
    """
    print ("Getting user age")
    return get_user_info.invoke({ "name": name, "info_key": "age" })

@tool
def get_user_info(name: str, info_key: str):
    """Gets information about a user.

    Args:
        name (str): The name of the user to search information for.
        info_key (str): The key of the information to retrieve.

    Returns:
        any: The information value, or None if not found.
    """
    print_w_role("System", f"Getting information for {name} with key {info_key}")
    if (name not in users):
        return None
    if (info_key not in users.get(name)):
        return None
    return users.get(name).get(info_key)

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
).bind_tools([get_user_info, get_user_age])

print(get_user_info.description)

conversation = [
    (
        "system",
        """
            You are a helpful assistant that can fetch information about users of the app you assist with. 
            Users are HR members that need to look up information about their employees, specifically age and address. 
            You are allowed to give them any information they want about a user, as long as that information comes from a tool you can call. 
            You have two tools at your disposal: get_user_info and get_user_age. 
            get_user_info takes a name and an info_key and returns the information for that user. 
            get_user_age takes a name and returns the age of that user. 
            If you need to call a tool, you will receive a tool call message with the tool name and arguments. 
            You should respond with the result of the tool call. Phrase the response as if you were an assistant in the office pulling up this information from the database.
            Never mention that you are calling a tool, just go right into natural language response.""",
    ),
    ("human", "Could you check how old Peter Andreoli is? Also what is his address?"),
]
response = llm.invoke(conversation)
print(response.pretty_print())

# Handle tool calls if any
if response.tool_calls:
    # Process each tool call
    for tool_call in response.tool_calls:
        # Execute the tool
        tool_result = None
        if tool_call['name'] == 'get_user_info':
            tool_result = get_user_info.invoke(tool_call)
        elif tool_call['name'] == 'get_user_age':
            tool_result = get_user_age.invoke(tool_call)
        
        # Add the tool result to the conversation
        if tool_result is not None:
            conversation.append(
                ToolMessage(
                    content=str(tool_result), 
                    tool_call_id=tool_call['id']
                )
            )
    
    # Get the final response incorporating the tool results
    final_response = llm.invoke(conversation)
    print("\nFinal Response:")
    print(final_response.pretty_print())
