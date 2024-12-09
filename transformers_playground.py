
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def my_addition_function(param1: int, param2: int):
    return f"The answer is, of course, {param1 + param2}!"

def generate_message(role: str, content: str):
    return {
        "role": role,
        "content": content,
    }


model_id = "meta-llama/Llama-3.1-8B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

tokenizer.chat_template = """{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' ~ message['role'] ~ '<|end_header_id|>\n\n' ~ message['content'] | trim ~ '<|eot_id|>' %}{% if loop.first %}{% set content = bos_token ~ content %}{% endif %}{{ content }}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"""  
print(f"Tokenizer Chat Template: {tokenizer.chat_template}")

model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)
device = "cpu"

messages = [
    { "role": "system", "content": 'You are a helpful assistant,answer in JSON with key "messages"' },
    { "role": "user", "content": "Who are you?" },
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generate_prompt=True,
    return_tensors="pt",
).to(model.device)

# Create an attention mask
# attention_mask = input_ids.ne(tokenizer.pad_token_id).long()


terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
output = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.1,
    pad_token_id=tokenizer.eos_token_id,
    return_dict_in_generate=True,
    # attention_mask=attention_mask,
    return_legacy_cache=True,
    # num_return_sequences=2,  # Ensure only one sequence is generated
)

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

messages = [
    {
        "role": "system",
        "content": f"""You are a helpful assistant with access to the following functions: 
{str(functions_metadata)}

To use these functions respond with:
<functioncall> {{"name":"function_name","arguments":{{"arg_1":"value_1","arg_2":"value_2",...}}}} </functioncall>

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
        "content": """<function_response> {"answer":3} </function_response>""",
    },
    # {"role": "user", "content": "What is 2 + 5?"},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt =True ,
    return_tensors = "pt"
).to(model.device)
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]
outputs = model.generate(
    input_ids,
    max_new_tokens = 256,
    eos_token_id = terminators,
    do_sample = True,
    temperature = 0.6,
    top_p = 0.9,
)
response = outputs[0][input_ids.shape[-1]:]
llm_responses = tokenizer.decode(response,skip_special_tokens=True)

print("--- START OF CONVERSATION ---")
print(llm_responses)
print("--- END OF CONVERSATION ---")