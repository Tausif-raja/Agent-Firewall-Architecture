from collections import defaultdict
import time

class StateManager:
    def __init__(self):
        # Stores history of prompts per user/session
        # Structure: {user_id: [prompt1, prompt2, ...]}
        self.history = defaultdict(list)
        
        # Stores total cost per user/session
        # Structure: {user_id: total_cost_float}
        self.costs = defaultdict(float)

    def add_to_history(self, user_id: str, prompt: str):
        self.history[user_id].append(prompt)
        # Keep only last 10 prompts to save memory and relevance
        if len(self.history[user_id]) > 10:
            self.history[user_id].pop(0)

    def get_history(self, user_id: str) -> list[str]:
        return self.history[user_id]

    def add_cost(self, user_id: str, cost: float):
        self.costs[user_id] += cost

    def get_cost(self, user_id: str) -> float:
        return self.costs[user_id]

    def clear_state(self, user_id: str):
        if user_id in self.history:
            del self.history[user_id]
        if user_id in self.costs:
            del self.costs[user_id]
