from litellm import completion
import os
from database.database import get_db
from typing import Optional
import time
from sqlalchemy.orm import Session


class Define_Agent():
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_KEY") or os.getenv("OPENROUTER_API_KEY")

    # def call(self, 
    #          model_name:str,
    #          message_dict:dict, 
    #          streamed:Optional[bool] = False,

    #          ):
    #     response = completion(
    #     model=f"{model_name}", 
    #     messages=message_dict,
    #     stream=streamed
    #     )
    #     return response

    def chat(self, user_id: str, messages: list, model: str = "openrouter/openai/gpt-4o-mini", db: Session = get_db(), **kwargs):
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
            response = completion(
                model=model,
                messages=messages,
                api_key= self.api_key,
                **kwargs
            )
            
            end_time = time.time()
            
            # if not kwargs.get('stream', False):
            #     self.log_request(db, user_id, model, messages, response, start_time, end_time)
            
            return response
        except Exception as e:
            print(f"🔥 Error: {e}")
            raise e
        finally:
            if close_session:
                db.close()
# Text
# messages=[{"role": "user", "content": [{"type": "text", "text": "Hello!"}]}]

# # Image
# messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}]}]

