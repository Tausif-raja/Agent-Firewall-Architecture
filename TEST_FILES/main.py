import os
import uvicorn
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from litellm import completion, completion_cost
from dotenv import load_dotenv

# Import our custom modules
from prompt_injection.analyzer import LoopDetector
from prompt_injection.state import StateManager
from prompt_injection.red_team import RedTeamer

# Load environment variables
load_dotenv()

app = FastAPI()

# Global State & Models
# In a production app, use Dependency Injection or a Singleton pattern
print("Initializing Firewall Components...")
loop_detector = LoopDetector()
state_manager = StateManager()
red_teamer = RedTeamer()
print("Firewall Ready.")

# Configuration
FORBIDDEN_WORDS = ["ignore instructions", "terminate", "system override", "pwned"]
MAX_LOOP_SIMILARITY_THRESHOLD = 0.95
COST_LIMIT_PER_USER = 50.0 # $50 limit

@app.post("/v1/chat/completions")
async def proxy_chat(request: Request, background_tasks: BackgroundTasks):
    """
    The Smart Proxy:
    1. Checks for Infinite Loops
    2. Checks for Cost Limits
    3. Checks for Basic Keyword Attacks
    4. Forwards to LLM (via LiteLLM)
    5. Updates State (History & Cost)
    """
    try:
        body = await request.json()
        messages = body.get("messages", [])
        
        # User ID is essential for tracking state. 
        # For this demo, we can use a header or default to "default_user"
        user_id = request.headers.get("x-user-id", "default_user")
        
        if not messages:
            raise HTTPException(status_code=400, detail="No messages provided")

        last_user_message = messages[-1]["content"] if messages[-1]["role"] == "user" else ""
        
        # --- LAYER 1: COST LIMIT ---
        current_spend = state_manager.get_cost(user_id)
        if current_spend >= COST_LIMIT_PER_USER:
             raise HTTPException(status_code=402, detail=f"Cost limit reached: ${current_spend}")

        # --- LAYER 2: KEYWORD FIREWALL ---
        for word in FORBIDDEN_WORDS:
            if word in last_user_message.lower():
                print(f"🛑 BLOCKED ATTACK: Found '{word}'")
                raise HTTPException(status_code=400, detail=f"Security Alert: Forbidden content '{word}'")

        # --- LAYER 3: LOOP DETECTION ---
        # Get recent history for this user
        history = state_manager.get_history(user_id)
        
        # Check similarity
        similarity = loop_detector.detect_loop(history, last_user_message)
        print(f"🔍 Loop Score: {similarity:.4f} (User: {user_id})")
        
        if similarity > MAX_LOOP_SIMILARITY_THRESHOLD:
            print(f"🛑 LOOP DETECTED for {user_id}. Blocking.")
            raise HTTPException(status_code=400, detail="Infinite Loop Detected. Your requests are too repetitive.")

        # --- LAYER 4: FORWARD TO LLM ---
        print(f"✅ PASSED: Sending to {body.get('model', 'gpt-3.5-turbo')}")
        
        # We need to correctly handle the response from LiteLLM
        # For simplicity in this demo, we assume non-streaming
        response = completion(
            model=body.get("model", "gpt-3.5-turbo"),
            messages=messages
        )
        
        # --- LAYER 5: UPDATE STATE (Async) ---
        # Calculate cost
        try:
            cost = completion_cost(completion_response=response)
            state_manager.add_cost(user_id, cost)
            print(f"💰 Cost detected: ${cost:.6f}")
        except:
             pass # Don't fail if cost calc fails
             
        # Add to history
        state_manager.add_to_history(user_id, last_user_message)

        return response

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        print(f"Error: {e}")
        return {"error": str(e)}

@app.post("/scan")
async def scan_agent(request: Request):
    """
    Triggers a Red Team scan against a specific URL.
    Usage: POST /scan {"target_url": "http://localhost:8000/v1/chat/completions"}
    """
    body = await request.json()
    target_url = body.get("target_url")
    if not target_url:
        raise HTTPException(status_code=400, detail="target_url is required")
    
    print(f"🚀 Launching Red Team attack on {target_url}")
    report = await red_teamer.scan(target_url)
    return report

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)