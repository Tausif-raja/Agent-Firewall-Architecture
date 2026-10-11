import os
import time
import uuid
from datetime import datetime
from litellm import completion, cost_calculator
from sqlalchemy.orm import Session
from Agent_Firewall.database.database import SessionLocal
from Agent_Firewall.database.models import RequestLog, ContentLog

class AgentGuard:
    def __init__(self):
        pass

    def log_request(self, db: Session, user_id, model, messages, response, start_time, end_time):
        """
        Calculates cost and logs the request/response to the database.
        """
        try:
            # 1. Calculate Cost
            try:
                # Prepare completion_response for cost calculation
                # cost_calculator.completion_cost expects the full response object or dict
                # If the user passed just 'model_name', help litellm out by inferring provider
                cost_model = model
                if "groq" not in cost_model and "llama" in cost_model:
                     cost_model = f"groq/{cost_model}"
                
                completion_cost = cost_calculator.completion_cost(
                    completion_response=response,
                    model=cost_model
                )
            except Exception as e:
                print(f"⚠️ Cost calculation failed: {e}")
                completion_cost = 0.0
            
            # 2. Extract Token Usage
            input_tokens = 0
            output_tokens = 0
            if hasattr(response, 'usage'):
                # Handle object access (e.g. ModelResponse object)
                input_tokens = getattr(response.usage, 'prompt_tokens', 0)
                output_tokens = getattr(response.usage, 'completion_tokens', 0)
            elif isinstance(response, dict) and 'usage' in response:
                 input_tokens = response['usage'].get('prompt_tokens', 0)
                 output_tokens = response['usage'].get('completion_tokens', 0)

            # 3. Create Request ID
            req_id = str(uuid.uuid4())

            # 4. Create RequestLog Entry
            log_entry = RequestLog(
                request_id=req_id,
                user_id=user_id,
                timestamp=datetime.utcnow(),
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_cost=completion_cost,
                response_status="success"
            )
            
            # 5. Create ContentLog Entry
            # Extract last user message and bot response
            last_user_msg = ""
            for m in reversed(messages):
                if m.get('role') == 'user':
                    last_user_msg = m.get('content', "")
                    break
            
            bot_content = ""
            if hasattr(response, 'choices') and len(response.choices) > 0:
                 bot_content = response.choices[0].message.content
            elif isinstance(response, dict) and 'choices' in response and len(response['choices']) > 0:
                 bot_content = response['choices'][0]['message']['content']

            content_entry = ContentLog(
                request_id=req_id,
                user_query=str(last_user_msg),
                bot_response=str(bot_content)
            )

            # 6. Save to DB
            db.add(log_entry)
            db.add(content_entry)
            db.commit()
            
            print(f"✅ [LOGGED] Cost: ${completion_cost:.5f} | User: {user_id}")
            
        except Exception as e:
            print(f"❌ Logging Failed: {e}")
            db.rollback()

    def chat(self, user_id: str, messages: list, model: str = "groq/llama-3.1-8b-instant", db: Session = None, **kwargs):
        """
        The Main Entry Point.
        Accepts generic parameters (**kwargs) to support various LLMs.
        """
        print(f"🔒 AgentGuard: Intercepting request for {user_id}...")
        
        start_time = time.time()
        
        # Use provided session or create a new one
        close_session = False
        if db is None:
            db = SessionLocal()
            close_session = True
        
        try:
            # 1. Call the LLM (Unified Proxy via LiteLLM)
            # Pass **kwargs to completion to support all provider-specific params
            response = completion(
                model=model,
                messages=messages,
                api_key=os.getenv("GROQ_API_KEY"),
                **kwargs
            )
            
            end_time = time.time()
            
            # 2. Log the result
            # Note: Streaming support would require handling chunks. 
            # For now, we assume stream=False or handle it separately if requested.
            # If kwargs.get('stream') is True, logging needs to happen after consuming stream
            if not kwargs.get('stream', False):
                self.log_request(db, user_id, model, messages, response, start_time, end_time)
            
            # 3. Return the Standard Object to the user
            return response

        except Exception as e:
            print(f"🔥 Error: {e}")
            # Log failure if possible
            raise e
        finally:
            if close_session:
                db.close()
