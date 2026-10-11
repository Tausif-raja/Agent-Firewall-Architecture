from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from database import get_db, engine, Base
import models as models

# Create Tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI Firewall Proxy")

# guard = AgentGuard()

# class ChatRequest(BaseModel):
#     user_id: str
#     model: str = "groq/llama-3.1-8b-instant"
#     messages: List[Dict[str, str]]
#     temperature: Optional[float] = 0.7
#     # Allow extra fields for other LLM params
#     extra_params: Optional[Dict[str, Any]] = {}

# @app.post("/chat")
# def chat_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
#     try:
#         # Merge explicitly defined params with extra_params
#         kwargs = request.extra_params or {}
#         if request.temperature is not None:
#              kwargs['temperature'] = request.temperature

#         response = guard.chat(
#             user_id=request.user_id,
#             messages=request.messages,
#             model=request.model,
#             db=db,
#             **kwargs
#         )
#         return response
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}
