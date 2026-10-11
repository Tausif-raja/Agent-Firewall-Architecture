from litellm import completion,completion_cost
from litellm import embedding

import os

from pathlib import Path
from dotenv import load_dotenv

# Keys live in Agent_Firewall/.env (never commit them). See .env.example.
load_dotenv(Path(__file__).resolve().parents[1] / "Agent_Firewall" / ".env")

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
