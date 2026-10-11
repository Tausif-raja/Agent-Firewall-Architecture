from litellm import completion, model_cost
import os
import re
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_KEY")

class FirewallRouter:
    def __init__(self):
        self.api_key = OPENROUTER_API_KEY
        
        # Define model tiers for cost and complexity routing
        self.models = {
            "fast_cheap": "openrouter/openai/gpt-3.5-turbo",
            "balanced": "openrouter/anthropic/claude-3-haiku",
            "powerful": "openrouter/openai/gpt-4o",
            "fallback": ["openrouter/anthropic/claude-3-haiku", "openrouter/google/gemini-pro"]
        }
        
        # Max budget per request (for cost-based routing simulation)
        self.MAX_BUDGET = 0.05 

    def _execute_completion(self, model: str, messages: list, fallbacks: list = None, **kwargs):
        """Helper to execute completion with optional fallbacks."""
        if fallbacks:
            kwargs["fallbacks"] = [{"model": f} for f in fallbacks]
            
        return completion(
            model=model,
            messages=messages,
            api_key=self.api_key,
            **kwargs
        )

    # 1. Model Selection Routing
    def route_specific_model(self, messages: list, model_name: str, provider_name: str = "openai"):
        """Route request to directly specific model"""
        target_model = f"openrouter/{provider_name}/{model_name}"
        return self._execute_completion(target_model, messages)

    # 2. Rule-Based Routing
    def route_by_rule(self, messages: list, context_length: int = 0):
        """If/else based routing based on content properties."""
        if context_length > 8000:
            target_model = "openrouter/anthropic/claude-3-sonnet"
        else:
            target_model = self.models["fast_cheap"]
            
        return self._execute_completion(target_model, messages)

    # 3. Cost-Based Routing (VERY IMPORTANT)
    def route_by_cost(self, messages: list, max_budget: float = None):
        """Route based on budget and cost-efficiency."""
        budget = max_budget or self.MAX_BUDGET
        
        if budget >= 0.03:
            model = self.models["powerful"]
        elif budget >= 0.01:
            model = self.models["balanced"]
        else:
            model = self.models["fast_cheap"]
            
        return self._execute_completion(model, messages)

    # 4. Fallback Routing
    def route_with_fallback(self, messages: list, primary_model: str = None):
        """If model fails -> switch."""
        primary = primary_model or self.models["powerful"]
        return self._execute_completion(primary, messages, fallbacks=self.models["fallback"])

    # 7. AI-Based Router
    def ai_based_route(self, messages: list):
        """Use a small model to decide: Which model should handle this? Dynamic + adaptive."""
        user_prompt = messages[-1]["content"] if isinstance(messages, list) and len(messages) > 0 else str(messages)
        
        eval_messages = [
            {"role": "system", "content": "Analyze the user prompt. Reply with EXACTLY ONE word indicating complexity: MATH, CODE, CREATIVE, or GENERAL."},
            {"role": "user", "content": f"Prompt: {user_prompt}"}
        ]
        
        decision_response = self._execute_completion(self.models["fast_cheap"], eval_messages)
        decision = decision_response.choices[0].message.content.strip().upper()
        
        if "CODE" in decision or "MATH" in decision:
            target_model = self.models["powerful"]
        elif "CREATIVE" in decision:
            target_model = self.models["balanced"]
        else:
            target_model = self.models["fast_cheap"]
            
        return self._execute_completion(target_model, messages)

    # 8. Prompt Complexity Routing
    def route_by_complexity(self, messages: list):
        """Detect complexity level through heuristics."""
        user_prompt = messages[-1]["content"] if isinstance(messages, list) and len(messages) > 0 else str(messages)
        
        length = len(user_prompt.split())
        code_markers = len(re.findall(r"```", user_prompt))
        complex_keywords = any(word in user_prompt.lower() for word in ["analyze", "synthesize", "architect", "quantum"])
        
        if length > 500 or code_markers > 0 or complex_keywords:
            target_model = self.models["powerful"]
        elif length > 100:
            target_model = self.models["balanced"]
        else:
            target_model = self.models["fast_cheap"]
            
        return self._execute_completion(target_model, messages)

def route_to_model(messages: str, model_name: str = "gpt-3.5-turbo", provider_name: str = "openai"):
    router = FirewallRouter()
    msgs = [{"role": "user", "content": messages}] if isinstance(messages, str) else messages
    return router.route_specific_model(msgs, model_name, provider_name)

if __name__ == "__main__":
    result = route_to_model("Hi How are you")
    print("this is result : ", result)

