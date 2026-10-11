from litellm import completion,completion_cost
from litellm import embedding

import os

os.environ['GROQ_API_KEY'] = "***REMOVED_GROQ_KEY***"

response = completion(
    model="groq/llama-3.1-8b-instant", 
    messages=[
        {"role":"system","content":"You are a helper in hospital front desk named Shila."},
       {"role": "user", "content": "hello from litellm, how are you?"}
   ],
)
# print("this is response : ", response)

# response = embedding(model='text-embedding-ada-002', input=["good morning from litellm"])
# print("this is response : ", response)
cost = completion_cost(completion_response=response)
formatted_string = f"${float(cost):.10f}"
print(formatted_string)
