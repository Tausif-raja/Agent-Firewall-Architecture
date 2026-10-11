import os
import sys
import json
import uuid
import secrets
from datetime import datetime, date
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

# Load environment variables (.env) from current directory and parent directory
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# Set standard OPENROUTER_API_KEY environment variable for LiteLLM if present
if os.getenv("OPENROUTER_KEY") and not os.getenv("OPENROUTER_API_KEY"):
    os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_KEY")

from fastapi import FastAPI, Depends, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging

logger = logging.getLogger("agent_firewall.main")

from Agent_Firewall.database.database import get_db, engine, Base, SessionLocal
from Agent_Firewall.database.models import (
    RequestLog,
    ContentLog,
    MetricsLog,
    DailyStats,
    ToolExecutionLog,
    ApiKeyConfig,
    utcnow,
)
from Agent_Firewall.firewall_engine import firewall_engine
from Agent_Firewall.security import RateLimiter, display_key, hash_api_key, migrate_plaintext_keys

# Ensure tables are created, and hash any keys stored in plaintext by older versions
Base.metadata.create_all(bind=engine)
with SessionLocal() as _db:
    _migrated = migrate_plaintext_keys(_db, ApiKeyConfig)
    if _migrated:
        logger.warning(f"Migrated {_migrated} plaintext API key(s) to hashed storage")

rate_limiter = RateLimiter()

# -------------------------------------------------------------
# Authentication
# -------------------------------------------------------------
# Admin token protects the dashboard data and management APIs. If it is not
# configured, a random one is generated per process and printed once.
ADMIN_TOKEN = os.getenv("FIREWALL_ADMIN_TOKEN", "").strip()
if not ADMIN_TOKEN:
    ADMIN_TOKEN = secrets.token_urlsafe(24)
    print(f"\n[Agent Firewall] FIREWALL_ADMIN_TOKEN not set. Generated admin token for this run:\n    {ADMIN_TOKEN}\n"
          f"Set FIREWALL_ADMIN_TOKEN in Agent_Firewall/.env to keep it stable.\n", file=sys.stderr, flush=True)


def require_admin(x_admin_token: Optional[str] = Header(None)) -> None:
    if not x_admin_token or not secrets.compare_digest(x_admin_token, ADMIN_TOKEN):
        raise HTTPException(status_code=401, detail="Admin token required (X-Admin-Token header)")


def _bearer(authorization: Optional[str]) -> Optional[str]:
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip() or None
    return None



app = FastAPI(
    title="Agent Firewall Security Proxy & Dashboard",
    description="Intelligent AI Agent Firewall with Prompt Injection Shield, Dual-Gate Tool Guard & Kill-Switch",
    version="2.0.0"
)

# Browsers may only call the API cross-origin from explicitly configured origins.
_cors_origins = [o.strip() for o in os.getenv("FIREWALL_CORS_ORIGINS", "").split(",") if o.strip()]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "X-Admin-Token"],
    )


# -------------------------------------------------------------
# Request & Response Schemas
# -------------------------------------------------------------
class CreateKeyRequest(BaseModel):
    key_name: str
    model: str = "openrouter/openai/gpt-4o-mini"
    provider: str = "openrouter"
    prompt_injection_enabled: bool = True
    tool_guard_enabled: bool = True
    kill_switch_enabled: bool = True
    sensitive_data_enabled: bool = True
    gdpr_mode: str = "tokenize"  # 'tokenize', 'mask', 'block'
    rate_limit_rpm: int = 60


class ChatMessage(BaseModel):
    # Keep tool_calls / tool_call_id / name etc. so tool-calling conversations survive the proxy.
    model_config = ConfigDict(extra="allow")
    role: str
    content: Any = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False
    session_id: Optional[str] = "default_session"
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None


# OpenAI parameters passed through to the upstream model unchanged.
FORWARDED_PARAMS = ("tools", "tool_choice", "parallel_tool_calls", "top_p", "stop", "response_format",
                    "seed", "presence_penalty", "frequency_penalty", "n", "logit_bias", "user")


def _message_dict(m: ChatMessage) -> Dict[str, Any]:
    d = m.model_dump(exclude_none=True)
    d.setdefault("content", None)
    return d


class PlaygroundChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: str = "openrouter/openai/gpt-4o-mini"
    provider_api_key: Optional[str] = None
    firewall_key: Optional[str] = None
    enable_prompt_injection: bool = True
    enable_tool_guard: bool = True
    enable_kill_switch: bool = True
    enable_sensitive_data: bool = True
    gdpr_mode: str = "tokenize"  # 'tokenize', 'mask', 'block'
    session_id: Optional[str] = "playground_session"
    simulate_tool_call: Optional[Dict[str, Any]] = None


class ToolSimulationRequest(BaseModel):
    tool_name: str
    tool_args: Dict[str, Any]
    user_role: str = "standard"
    session_id: str = "playground_session"


# -------------------------------------------------------------
# Supported Models Catalog
# -------------------------------------------------------------
SUPPORTED_MODELS = [
    {
        "id": "openrouter/openai/gpt-4o-mini",
        "name": "GPT-4o Mini",
        "provider": "OpenAI / OpenRouter",
        "speed": "Ultra Fast (Sub-200ms)",
        "pricing": "$0.15 / 1M tokens",
        "recommended": True
    },
    {
        "id": "openrouter/deepseek/deepseek-r1",
        "name": "DeepSeek R1 (Reasoning)",
        "provider": "DeepSeek / OpenRouter",
        "speed": "Thinking Model",
        "pricing": "$0.55 / 1M tokens",
        "recommended": True
    },
    {
        "id": "openrouter/deepseek/deepseek-chat",
        "name": "DeepSeek V3",
        "provider": "DeepSeek / OpenRouter",
        "speed": "Ultra Fast",
        "pricing": "$0.14 / 1M tokens",
        "recommended": False
    },
    {
        "id": "openrouter/openai/o3-mini",
        "name": "OpenAI o3-mini (Reasoning)",
        "provider": "OpenAI / OpenRouter",
        "speed": "Fast Reasoning",
        "pricing": "$1.10 / 1M tokens",
        "recommended": True
    },
    {
        "id": "openrouter/meta-llama/llama-3.3-70b-instruct",
        "name": "Llama 3.3 70B Instruct",
        "provider": "Meta / OpenRouter",
        "speed": "Fast (128K context)",
        "pricing": "$0.22 / 1M tokens",
        "recommended": False
    },
    {
        "id": "openrouter/qwen/qwen-2.5-coder-32b-instruct",
        "name": "Qwen 2.5 Coder 32B",
        "provider": "Qwen / OpenRouter",
        "speed": "Fast (Code & Agents)",
        "pricing": "$0.07 / 1M tokens",
        "recommended": False
    },
    {
        "id": "openrouter/mistralai/mistral-large-2407",
        "name": "Mistral Large",
        "provider": "Mistral AI / OpenRouter",
        "speed": "Balanced Enterprise",
        "pricing": "$2.00 / 1M tokens",
        "recommended": False
    }
]


# -------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------
@app.get("/api/models")
def get_supported_models():
    """Returns catalog of supported AI models and providers."""
    return {"models": SUPPORTED_MODELS}


@app.post("/api/keys/create", dependencies=[Depends(require_admin)])
def create_api_key(req: CreateKeyRequest, db: Session = Depends(get_db)):
    """
    Generates a virtual Firewall API Key associated with a specific model and security policy.
    """
    # Generate unique key prefix: agf-live-...
    random_token = secrets.token_hex(16)
    generated_key = f"agf-live-{random_token}"

    key_record = ApiKeyConfig(
        key_name=req.key_name.strip() or "Default Key",
        api_key=hash_api_key(generated_key),  # only the hash is stored
        model=req.model,
        provider=req.provider,
        created_at=utcnow(),
        is_active=1,
        prompt_injection_enabled=1 if req.prompt_injection_enabled else 0,
        tool_guard_enabled=1 if req.tool_guard_enabled else 0,
        kill_switch_enabled=1 if req.kill_switch_enabled else 0,
        sensitive_data_enabled=1 if req.sensitive_data_enabled else 0,
        gdpr_mode=req.gdpr_mode or "tokenize",
        rate_limit_rpm=req.rate_limit_rpm,
        total_calls=0
    )
    db.add(key_record)
    db.commit()
    db.refresh(key_record)

    return {
        "success": True,
        "api_key": generated_key,  # shown once; cannot be recovered later
        "key_name": key_record.key_name,
        "model": key_record.model,
        "provider": key_record.provider,
        "proxy_endpoint": "/api/v1/chat/completions",
        "created_at": key_record.created_at.isoformat()
    }


@app.get("/api/keys", dependencies=[Depends(require_admin)])
def list_api_keys(db: Session = Depends(get_db)):
    """Lists all configured virtual Firewall API keys."""
    keys = db.query(ApiKeyConfig).order_by(ApiKeyConfig.created_at.desc()).all()
    return {
        "keys": [
            {
                "id": k.id,
                "key_name": k.key_name,
                "api_key": display_key(k.api_key),  # only the hash is stored
                "model": k.model,
                "provider": k.provider,
                "created_at": k.created_at.strftime("%Y-%m-%d %H:%M") if k.created_at else "",
                "is_active": bool(k.is_active),
                "prompt_injection_enabled": bool(k.prompt_injection_enabled),
                "tool_guard_enabled": bool(k.tool_guard_enabled),
                "kill_switch_enabled": bool(k.kill_switch_enabled),
                "sensitive_data_enabled": bool(getattr(k, "sensitive_data_enabled", 1)),
                "gdpr_mode": getattr(k, "gdpr_mode", "tokenize") or "tokenize",
                "rate_limit_rpm": k.rate_limit_rpm,
                "total_calls": k.total_calls or 0
            }
            for k in keys
        ]
    }


@app.delete("/api/keys/{key_id}", dependencies=[Depends(require_admin)])
def delete_api_key(key_id: int, db: Session = Depends(get_db)):
    """Deletes or revokes a firewall API key."""
    record = db.query(ApiKeyConfig).filter(ApiKeyConfig.id == key_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="API Key not found")
    db.delete(record)
    db.commit()
    return {"success": True, "message": "API key revoked"}


@app.post("/api/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest,
    request: Request,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """
    OpenAI-compatible chat completion proxy endpoint.
    Agents & client apps set base_url='http://localhost:8000/api/v1' and api_key='agf-live-...'
    """
    firewall_key = _bearer(authorization)
    if not firewall_key:
        raise HTTPException(status_code=401, detail="Missing firewall API key (Authorization: Bearer agf-live-...)")
    key_record = db.query(ApiKeyConfig).filter(ApiKeyConfig.api_key == hash_api_key(firewall_key)).first()
    if not key_record or not key_record.is_active:
        raise HTTPException(status_code=401, detail="Invalid or inactive Firewall API Key")
    allowed, retry_after = rate_limiter.check(f"key{key_record.id}", key_record.rate_limit_rpm or 0)
    if not allowed:
        raise HTTPException(status_code=429, detail=f"Rate limit of {key_record.rate_limit_rpm} requests/minute exceeded",
                            headers={"Retry-After": str(retry_after)})

    target_model = req.model or key_record.model
    forwarded = {k: v for k, v in req.model_dump(exclude_none=True).items() if k in FORWARDED_PARAMS}

    result = firewall_engine.process_chat(
        user_id=key_record.key_name,
        tenant_id=f"key{key_record.id}",
        messages=[_message_dict(m) for m in req.messages],
        model=target_model,
        firewall_key=firewall_key,
        session_id=req.session_id,
        db=db,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        **forwarded
    )

    if result.get("status") == "BLOCKED":
        return JSONResponse(status_code=403, content=result)
    elif result.get("status") == "ERROR":
        code = 502 if str(result.get("error_type", "")).startswith("upstream") else 500
        return JSONResponse(status_code=code, content=result)

    if req.stream:
        return StreamingResponse(_as_sse(result["response"]), media_type="text/event-stream")
    return result.get("response")


def _as_sse(body: Dict[str, Any]):
    """Replay a completed (already inspected) response as OpenAI-style SSE chunks.

    The upstream call is not streamed: egress checks need the full reply before
    anything reaches the client, so the client receives it in one content chunk.
    """
    choice = body["choices"][0]
    msg = choice["message"]
    delta: Dict[str, Any] = {"role": "assistant", "content": msg.get("content")}
    if msg.get("tool_calls"):
        delta["tool_calls"] = [{"index": i, **tc} for i, tc in enumerate(msg["tool_calls"])]
    base = {"id": body["id"], "object": "chat.completion.chunk", "created": body["created"], "model": body["model"]}
    chunks = [
        {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
        {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}], "usage": body.get("usage")},
    ]
    if body.get("firewall"):
        chunks[-1]["firewall"] = body["firewall"]
    for chunk in chunks:
        yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/api/chat/playground", dependencies=[Depends(require_admin)])
def playground_chat(req: PlaygroundChatRequest, db: Session = Depends(get_db)):
    """
    Interactive Chat Playground:
    Allows user to test prompts, inspect live security pipeline decisions,
    measure latency, tokens, cost, and simulate tool calls.
    """
    messages_payload = [m.dict() for m in req.messages]
    last_prompt = messages_payload[-1]["content"] if messages_payload else ""

    pipeline_trace = []

    # Step 0: Sensitive Data & GDPR Shield
    sanitized_prompt = last_prompt
    sensitive_findings = []
    if req.enable_sensitive_data and last_prompt:
        sanitized_prompt, sensitive_findings, should_block_sens = firewall_engine.sensitive_engine.sanitize_or_pseudonymize(
            text=str(last_prompt),
            session_id=req.session_id,
            mode=req.gdpr_mode or "tokenize"
        )
        if should_block_sens:
            types = ", ".join(set(f.entity_type for f in sensitive_findings))
            pipeline_trace.append({
                "step": "Sensitive Data & GDPR Guard",
                "status": "BLOCKED",
                "detail": f"Unauthorized confidential data / GDPR violation detected: {types}",
                "latency_ms": 1.4
            })
            req_id = str(uuid.uuid4())
            blocked_msg = f"🔒 [Agent Firewall Sensitive Data Guard]: Blocked payload containing {types}."
            firewall_engine._log_transaction(
                db=db,
                req_id=req_id,
                user_id="playground_user",
                model=req.model,
                user_query=str(sanitized_prompt),
                bot_response=blocked_msg,
                status="blocked_sensitive_data",
                latency_ms=2.0,
                input_tokens=len(str(last_prompt).split()),
                output_tokens=10,
                cost=0.0
            )
            return {
                "success": False,
                "status": "BLOCKED",
                "response": blocked_msg,
                "pipeline_trace": pipeline_trace,
                "sensitive_findings": [f.to_dict() for f in sensitive_findings],
                "tokens": {"input": len(str(last_prompt).split()), "output": 10, "total": len(str(last_prompt).split()) + 10},
                "cost_usd": 0.0,
                "latency_ms": 2.0,
                "tool_result": None
            }
        elif sensitive_findings:
            types = ", ".join(sorted(set(f.entity_type for f in sensitive_findings)))
            mode_desc = "Pseudonymized with Reversible Token Vault" if req.gdpr_mode == "tokenize" else "Redacted"
            logger.warning(
                f"🚨 [Agent Firewall Playground]: Sensitive data detected & redacted ({types})! "
                f"Findings: {[f.to_dict() for f in sensitive_findings]}"
            )
            pipeline_trace.append({
                "step": "Microsoft Presidio & Sensitive Data Guard",
                "status": "UNSAFE (REDACTED)",
                "detail": f"Detected & redacted {len(sensitive_findings)} sensitive entities ({types}). Detailed logs recorded.",
                "latency_ms": 1.8
            })
            # Replace in payload for downstream LLM call
            if messages_payload:
                messages_payload[-1]["content"] = sanitized_prompt
        else:
            pipeline_trace.append({
                "step": "Sensitive Data & GDPR Guard",
                "status": "CLEAN",
                "detail": "No credentials, API keys, PII, or GDPR Art. 9 special attributes found.",
                "latency_ms": 0.9
            })
    else:
        pipeline_trace.append({
            "step": "Sensitive Data & GDPR Guard",
            "status": "BYPASS",
            "detail": "Sensitive data filter disabled by client toggle.",
            "latency_ms": 0.0
        })

    # Step 1: Prompt Inspection
    ingress_status = "PASSED"
    matched_detail = None
    if req.enable_prompt_injection:
        inspect_res = firewall_engine.inspect_prompt(str(last_prompt), check_sensitive=False)
        if not inspect_res.is_safe:
            ingress_status = "BLOCKED"
            matched_detail = f"{inspect_res.threat_type} (Pattern: '{inspect_res.matched_pattern}')"
            pipeline_trace.append({
                "step": "Prompt Injection Shield",
                "status": "BLOCKED",
                "detail": matched_detail,
                "latency_ms": 1.2
            })
        else:
            pipeline_trace.append({
                "step": "Prompt Injection Shield",
                "status": "SAFE",
                "detail": "Input verified clean. No injection or exfiltration patterns detected.",
                "latency_ms": 0.8
            })
    else:
        pipeline_trace.append({
            "step": "Prompt Injection Shield",
            "status": "BYPASS",
            "detail": "Filter disabled by client toggle.",
            "latency_ms": 0.0
        })

    # If blocked at ingress, immediately log and return
    if ingress_status == "BLOCKED":
        req_id = str(uuid.uuid4())
        blocked_msg = f"🛡️ [Agent Firewall Blocked Prompt]: {matched_detail}"
        firewall_engine._log_transaction(
            db=db,
            req_id=req_id,
            user_id="playground_user",
            model=req.model,
            user_query=str(last_prompt),
            bot_response=blocked_msg,
            status="blocked_injection",
            latency_ms=2.0,
            input_tokens=len(str(last_prompt).split()),
            output_tokens=10,
            cost=0.0
        )
        return {
            "success": False,
            "status": "BLOCKED",
            "response": blocked_msg,
            "pipeline_trace": pipeline_trace,
            "tokens": {"input": len(str(last_prompt).split()), "output": 10, "total": len(str(last_prompt).split()) + 10},
            "cost_usd": 0.0,
            "latency_ms": 2.0,
            "tool_result": None
        }

    # Step 2: Tool Guard Simulation (if requested)
    tool_exec_result = None
    if req.simulate_tool_call and req.enable_tool_guard:
        t_name = req.simulate_tool_call.get("name", "fetch_weather")
        t_args = req.simulate_tool_call.get("args", {})
        tool_res = firewall_engine.execute_tool_proxy(
            tool_name=t_name,
            tool_args=t_args,
            session_id=req.session_id,
            db=db
        )
        tool_exec_result = tool_res
        pipeline_trace.append({
            "step": "Tool Guard (Dual-Gate Proxy)",
            "status": "APPROVED" if tool_res.get("success") else "BLOCKED",
            "detail": f"Tool: '{t_name}' -> {tool_res.get('status')}. Gate: {tool_res.get('gate', 'Dual-Gate')}",
            "latency_ms": tool_res.get("latency_ms", 15.0)
        })

    # Step 3: LLM Execution
    chat_result = firewall_engine.process_chat(
        user_id="playground_user",
        messages=messages_payload,
        model=req.model,
        api_key=req.provider_api_key,
        firewall_key=req.firewall_key,
        session_id=req.session_id,
        db=db,
        tenant_id="playground"
    )

    bot_reply = ""
    if chat_result.get("response") and "choices" in chat_result["response"]:
        bot_reply = chat_result["response"]["choices"][0]["message"]["content"] or ""
    elif chat_result.get("error"):
        bot_reply = f"Error: {chat_result.get('error')}"

    # Step 4: Kill Switch Check
    if req.enable_kill_switch:
        loop_status = "SAFE"
        loop_detail = "No repetitive loop detected."
        if chat_result.get("status") == "LOOP_TERMINATED":
            loop_status = "TERMINATED"
            loop_detail = "Kill-Switch intercepted repetitive agent loop."

        pipeline_trace.append({
            "step": "Kill-Switch Loop Detector",
            "status": loop_status,
            "detail": loop_detail,
            "latency_ms": 2.5
        })

    # Safe unmasked preview for the authorized sender if tokenized
    unmasked_response = bot_reply
    if req.enable_sensitive_data and req.gdpr_mode == "tokenize" and bot_reply:
        unmasked_response = firewall_engine.sensitive_engine.unmask_egress(bot_reply, session_id=req.session_id)

    final_status = "UNSAFE" if sensitive_findings else chat_result.get("status", "SUCCESS")
    final_verdict = "UNSAFE" if sensitive_findings else ("SAFE" if final_status == "SUCCESS" else "TERMINATED")

    return {
        "success": chat_result.get("status") in ["SUCCESS", "LOOP_TERMINATED", "UNSAFE"],
        "status": final_status,
        "verdict": final_verdict,
        "response": bot_reply,
        "redacted_prompt": sanitized_prompt if sensitive_findings else None,
        "unmasked_response": unmasked_response,
        "pipeline_trace": pipeline_trace,
        "sensitive_findings": [f.to_dict() for f in sensitive_findings],
        "ropa_record": firewall_engine.sensitive_engine.generate_ropa_record(sensitive_findings, session_id=req.session_id) if sensitive_findings else None,
        "tokens": chat_result.get("tokens", {"input": 0, "output": 0, "total": 0}),
        "cost_usd": chat_result.get("cost_usd", 0.0),
        "latency_ms": chat_result.get("latency_ms", 0.0),
        "model_used": chat_result.get("model_used", req.model),
        "tool_result": tool_exec_result
    }


@app.post("/api/tools/simulate", dependencies=[Depends(require_admin)])
def simulate_tool(req: ToolSimulationRequest, db: Session = Depends(get_db)):
    """Simulates a tool call against the Dual-Gate Tool Guard."""
    res = firewall_engine.execute_tool_proxy(
        tool_name=req.tool_name,
        tool_args=req.tool_args,
        user_role=req.user_role,
        session_id=req.session_id,
        db=db
    )
    return res


@app.get("/api/dashboard/stats", dependencies=[Depends(require_admin)])
def get_dashboard_stats(db: Session = Depends(get_db)):
    """Aggregates metrics and statistics for the dashboard cards."""
    total_requests = db.query(RequestLog).count()
    blocked_count = db.query(RequestLog).filter(RequestLog.status.like("%blocked%")).count()
    success_count = db.query(RequestLog).filter(RequestLog.status == "success").count()

    total_tokens = 0
    total_cost = 0.0
    metrics_records = db.query(MetricsLog).all()
    for m in metrics_records:
        total_tokens += (m.total_tokens or 0)
        total_cost += (m.total_cost or 0.0)

    # Average latency
    reqs = db.query(RequestLog).all()
    avg_latency = round(sum([r.latency_ms or 0 for r in reqs]) / max(len(reqs), 1), 1)

    active_keys = db.query(ApiKeyConfig).filter(ApiKeyConfig.is_active == 1).count()
    tool_calls_count = db.query(ToolExecutionLog).count()

    return {
        "total_requests": total_requests,
        "blocked_attacks": blocked_count,
        "successful_requests": success_count,
        "protection_rate": f"{round((blocked_count / max(total_requests, 1)) * 100, 1)}%",
        "total_tokens": total_tokens,
        "total_cost_usd": round(total_cost, 5),
        "avg_latency_ms": avg_latency,
        "active_api_keys": active_keys,
        "total_tool_executions": tool_calls_count
    }


@app.get("/api/dashboard/logs", dependencies=[Depends(require_admin)])
def get_dashboard_logs(limit: int = 50, db: Session = Depends(get_db)):
    """Returns chronologically ordered audit logs with tool executions."""
    logs = (
        db.query(RequestLog)
        .order_by(RequestLog.timestamp.desc())
        .limit(limit)
        .all()
    )

    result = []
    for r in logs:
        c = r.content_log
        m = r.metrics_log
        tools = [
            {
                "tool_name": t.tool_name,
                "tool_args": t.tool_args,
                "authorized": t.authorized,
                "block_reason": t.block_reason,
                "execution_status": t.execution_status,
                "latency_ms": t.latency_ms,
                "estimated_tokens": t.estimated_tokens,
                "output_snippet": t.output_snippet
            }
            for t in r.tool_logs
        ]

        result.append({
            "request_id": r.request_id,
            "user_id": r.user_id,
            "model": r.model,
            "timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M:%S") if r.timestamp else "",
            "latency_ms": round(r.latency_ms or 0, 1),
            "status": r.status,
            "user_query": c.user_query if c else "",
            "bot_response": c.bot_response if c else "",
            "tokens": {
                "input": m.input_tokens if m else 0,
                "output": m.output_tokens if m else 0,
                "total": m.total_tokens if m else 0
            },
            "cost_usd": round(m.total_cost if m else 0.0, 6),
            "tools": tools
        })

    return {"logs": result}


# -------------------------------------------------------------
# Frontend Dashboard HTML View
# -------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index_page():
    """Serves the dashboard HTML interface."""
    static_html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(static_html_path):
        with open(static_html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Agent Firewall Backend Running</h1><p>Static files loading...</p>"


# Mount static files directory if exists
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("Agent_Firewall.main:app", host="0.0.0.0", port=8000, reload=True)
