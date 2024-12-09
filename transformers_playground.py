from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

def my_addition_function(param1: int, param2: int):
    """
    Custom addition function that returns a formatted string.
    
    Args:
        param1 (int): First number to add
        param2 (int): Second number to add
    
    Returns:
        str: Formatted addition result
    """
    return f"bababahshshsh {param1 + param2}!"

def handle_function_call(function_name, arguments):
    """
    Dynamically handle function calls based on the function name.
    
    Args:
        function_name (str): Name of the function to call
        arguments (dict): Dictionary of arguments to pass to the function
    
    Returns:
        dict: Result of the function call or error information
    """
    # Create a mapping of function names to actual function references
    function_map = {
        "do_addition": my_addition_function
    }
    
    # Check if the function exists in our map
    if function_name not in function_map:
        return {"error": f"Function {function_name} not found"}
    
    # Call the function with unpacked arguments
    try:
        return {"answer": function_map[function_name](**arguments)}
    except Exception as e:
        return {"error": str(e)}

def process_messages(messages, model, tokenizer):
    """
    Process messages and handle potential function calls
    
    Args:
        messages (list): Conversation history
        model: The language model
        tokenizer: The tokenizer for the model
    
    Returns:
        str: Processed response
    """
    # Tokenize and generate model response
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    # Define token terminators for generation
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    # Generate model output
    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    
    # Decode the response
    response = outputs[0][input_ids.shape[-1]:]
    llm_response = tokenizer.decode(response, skip_special_tokens=True)
    
    # Check if the response contains a function call
    function_call_match = re.search(r'<functioncall>(.*?)</functioncall>', llm_response)
    
    if function_call_match:
        print(f"[System] Function call detected: {function_call_match.group(1)}")
        try:
            # Parse the function call
            function_call_str = function_call_match.group(1)
            function_call = eval(function_call_str)
            
            # Handle the function call
            function_result = handle_function_call(
                function_call['name'], 
                function_call['arguments']
            )
            
            # Add the function result to messages
            messages.append({
                "role": "user",
                "content": f"<function_response>{str(function_result)}</function_response>"
            })
            
            # Recursively process messages with the function result
            return process_messages(messages, model, tokenizer)
        
        except Exception as e:
            print(f"Error processing function call: {e}")
            return llm_response
    
    return llm_response

# Existing model and tokenizer setup
model_id = "meta-llama/Llama-3.1-8B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

tokenizer.chat_template = """{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' ~ message['role'] ~ '<|end_header_id|>\n\n' ~ message['content'] | trim ~ '<|eot_id|>' %}{% if loop.first %}{% set content = bos_token ~ content %}{% endif %}{{ content }}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"""

model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)

# Function metadata (same as before)
functions_metadata = [
    {
        "type": "function",
        "function": {
            "name": "do_addition",
            "description": "Add two numbers together",
            "parameters": [
                {
                    "name": "param1",
                    "type": "int",
                    "description": "The first number to add",
                    "required": True,
                },
                {
                    "name": "param2",
                    "type": "int",
                    "description": "The second number to add",
                    "required": True,
                },
            ],
        },
    }
]

# Prepare initial messages with system context
messages = [
    {
        "role": "system",
        "content": f"""You are a helpful assistant with access to the following functions: 
{str(functions_metadata)}

To use these functions respond with:
<functioncall> {{"name":"function_name","arguments":{{"arg_1":"value_1","arg_2":"value_2",...}}}} </functioncall>

Ensure the types of the arguments match the function's requirements.

Edge cases you must handle:
If there are no functions that match the user request, you will respond politely that you cannot help.""",
    },
    {"role": "user", "content": "What is 1 + 2?"},
    {
        "role": "assistant",
        "content": """<functioncall>{"name":"do_addition","arguments":{"param1":1, "param2":2}}</functioncall>""",
    },
    {
        "role": "user",
        "content": """<function_response> {"answer":"haha its 3"} </function_response>""",
    },
    {
        "role": "assistant",
        "content": """The answer is haha its 3""",
    },
]

# Run the message processing
result = process_messages(messages, model, tokenizer)
print("--- START OF CONVERSATION ---")
print(result)
print("--- END OF CONVERSATION ---")