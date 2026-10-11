# https://gemini.google.com/app/8a67b1afd7c6df30 - google


import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from values import mock_responses

class ExactRepeatFilter:
    """Layer A: The Exact Repeat Filter (Fast & Cheap)"""
    def __init__(self, window_size: int = 3):
        self.window_size = window_size

    def check(self, history: list[str]) -> bool:
        if len(history) < self.window_size:
            return False
        
        last_n = history[-self.window_size:]
        hashes = [hashlib.md5(s.encode('utf-8')).hexdigest() for s in last_n]
        
        # If all hashes in the window are exactly the same, it's a literal loop
        return all(h == hashes[0] for h in hashes)


class SemanticVectorCheck:
    """Layer B: The Semantic Vector Check (The Sweet Spot)"""
    def __init__(self, threshold: float = 0.96, window_size: int = 5):
        # Using the specified lightweight SentenceTransformer model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.threshold = threshold
        self.window_size = window_size

    def check(self, history: list[str]) -> bool:
        if len(history) < 3:
            return False
            
        # Get the recent window of responses
        recent = history[-self.window_size:]
        embeddings = self.model.encode(recent)
        
        # Compare the absolute latest response against previous ones in the window
        latest_emb = embeddings[-1].reshape(1, -1)
        prev_embs = embeddings[:-1]
        
        sims = cosine_similarity(latest_emb, prev_embs)[0]
        
        # If the similarity to PREVIOUS messages is > 0.96 multiple times, trigger kill switch
        high_sim_count = sum(1 for s in sims if s > self.threshold)
        return high_sim_count >= 2


class StepBackJudge:
    """Layer C: The Step-Back Judge (The Intelligent Layer)"""
    def __init__(self, model_name: str = "llama3-8b-8192"):
        # We use a fast & cheap model for evaluating via Groq
        self.llm = ChatGroq(model=model_name, temperature=0.0,api_key="--")
        self.prompt = PromptTemplate.from_template(
            "Review the following agent transcript.\n"
            "Transcript:\n{transcript}\n\n"
            "Is the agent repeating a logic error or stuck in a tool-call loop?"
            "Answer only YES or NO."
        )
        self.chain = self.prompt | self.llm

    def check(self, history: list[str], turn_count: int) -> bool:
        # Only trigger this judge every 5th message if session exceeds 10 turns
        if turn_count < 10 or turn_count % 5 != 0:
            return False
            
        transcript = "\n".join([f"Turn {i+1}: {msg}" for i, msg in enumerate(history)])
        
        try:
            response = self.chain.invoke({"transcript": transcript})
            content = response.content.strip().upper()
            return "YES" in content
        except Exception as e:
            print(f"Step back judge failed: {e}")
            return False


class LoopDetector:
    """
    Main Orchestrator for the 3-Layer Loop Detection system.
    Instantiate this and call it per session/agent interaction.
    """
    def __init__(self):
        self.layer_a = ExactRepeatFilter()
        self.layer_b = SemanticVectorCheck()
        self.layer_c = StepBackJudge()
        self.history = []
        self.turn_count = 0

    def add_response(self, response: str) -> dict:
        """
        Processes a new agent response through the firewall layers.
        Returns a dictionary indicating if a loop was detected and by which layer.
        """
        self.history.append(response)
        self.turn_count += 1
        
        # Run Layer A (O(1) complexity, near 0 latency)
        if self.layer_a.check(self.history):
            return {"loop_detected": True, "layer": "A", "reason": "Literal exact repetition detected in last 3 turns."}
            
        # Run Layer B (O(n) vector complexity, very low latency cache)
        if self.layer_b.check(self.history):
            return {"loop_detected": True, "layer": "B", "reason": "Semantic similarity > 0.96 across multiple recent turns."}
            
        # Run Layer C (LLM Call, medium latency but regulated to every 5th turn > 10)
        if self.layer_c.check(self.history, self.turn_count):
            return {"loop_detected": True, "layer": "C", "reason": "Step-Back LLM Judge determined agent is stuck."}
            
        return {"loop_detected": False, "layer": None, "reason": "OK"}

if __name__ == "__main__":
    from values import mock_responses
    
    print("Initializing Loop Detector...\n")
    detector = LoopDetector()
    
    print(f"Testing {len(mock_responses)} responses from values.py:\n")
    for i, resp in enumerate(mock_responses):
        print(f"--- Adding Response {i+1} ---")
        print(f"Snippet: {resp[:100]}...")
        
        result = detector.add_response(resp)
        print(f"Result: {result}\n")
        
        if result["loop_detected"]:
            print(f"🚨 Loop caught by Layer {result['layer']} at turn {i+1}")
            # If you wanted to stop processing on detection, break here.
            # break

