
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage
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
    return get_user_info(name, "age")

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

messages = [
    (
        "system",
        "You are a helpful assistant that can fetch information about users. ",
    ),
    ("human", "Could you check how old Peter Andreoli is? Also what is his address?"),
]
ai_msg = llm.invoke(messages)
print(ai_msg.pretty_print())
