# Agent Firewall Architecture

## Overview
The Agent Firewall acts as a robust middleware layer positioned between the user and the LLM (Large Language Model) or AI Agents. Its primary purpose is to inspect, filter, and secure interactions, mitigating risks associated with prompt injections, data leakage, and runaway agent loops.

## Core Components

1. **Prompt Injection Detection & Prevention** (full design: [docs/PROMPT_INJECTION.md](docs/PROMPT_INJECTION.md))
   - **Scope:** direct injection (user messages) **and** indirect injection (PDFs, web pages, RAG chunks, tool responses, sub-agent messages).
   - **Provenance:** all content is handled as labeled segments with a trust level (`SYSTEM` > `USER` > `SEMI` > `UNTRUSTED`). The trust level comes from how the content arrived, never from what it says.
   - **Detection cascade:**
     - **L0 Normalize:** Unicode/homoglyph/zero-width cleanup, decoding of encoded payloads.
     - **L1 Rules:** versioned signatures for low-cost known attacks.
     - **L2 Classifier:** a fine-tuned transformer (Llama Prompt Guard 2 / DeBERTa-v3 prompt-injection), the primary detector.
     - **L3 LLM judge:** a hardened judge that only sees gray-zone cases.
   - **Indirect defenses:** format-aware extractors that surface hidden text, scanning at ingest and on every tool result, spotlighting of untrusted content in the prompt, **session taint tracking with tool-call capability control**, and egress exfiltration checks.
   - Detection lowers the attack rate. Capability control limits the damage when detection misses.

2. **Agent Loop Management (Kill-Switch)** (`Agent_Firewall/firewall_engine.py`)
   - Fingerprints every reply as *text + proposed tool calls* and keeps a short history per **API key + session**, so tenants never share state.
   - **Layer A:** the same fingerprint three times in a row (identical text, or the same tool with the same arguments) terminates the loop.
   - **Layer B:** near-identical text replies across four turns. This applies to text only: calling the same tool with *different* arguments (e.g. iterating over a list) is progress, not a loop.
   - *Why not embeddings of reasoning:* tested on the project's own demo data, MiniLM similarity missed the stuck loop (max 0.87 < 0.96 threshold) and flagged a legitimate batch job. Reasoning tokens are also often hidden by providers.

3. **Sensitive Data, PII & GDPR** (`firewall/sensitive_data_engine.py`)
   - Detects credentials (provider API keys, passwords, connection strings), PII (email, phone, Luhn-valid cards, IBAN, SSN, IP) and GDPR Art. 9 special categories; Presidio is used additionally when installed.
   - Per-key mode: **tokenize** (reversible per-session vault; values restored for the sender), **mask**, or **block**. Credentials are always masked irreversibly.
   - Applies to every text part of the latest user message (including multimodal content). Replies never echo credentials. The audit log stores redacted text only.

4. **Tool Guard** (`firewall/tool_guard.py`)
   - **Gate 1:** allowlist, roles, argument validation, destructive-operation patterns (`DROP TABLE`, `rm -rf`, …), human approval for high-risk tools.
   - **Execution:** timeout. **Gate 2:** credentials stripped from tool output, output truncated.
   - Tool calls *proposed by the model* are screened before they reach the client agent: destructive arguments and destinations taken from untrusted content are dropped.

5. **Access Control** (`Agent_Firewall/main.py`, `Agent_Firewall/security.py`)
   - Agents authenticate with virtual keys (`agf-live-…`), stored only as SHA-256 hashes, each with its own model, protection policy and per-minute rate limit.
   - Dashboard and management APIs require an admin token. CORS is same-origin unless configured.

## Request Flow (`POST /api/v1/chat/completions`)
1. **Authenticate** the virtual key → 401. **Rate limit** → 429.
2. **Sensitive data:** tokenize/mask/block PII and credentials in the latest user message.
3. **Direct injection:** rule check on the latest user message → 403.
4. **Indirect injection:** every `role: "tool"` message is scanned as untrusted (rules, plus the ML classifier when installed). Injections are removed, other content is wrapped as data, and the model is told not to follow it. The session is marked tainted.
5. **LLM call** via LiteLLM, with `tools`/`tool_choice` forwarded. Upstream failures return 502 and are logged as failures.
6. **Tool-call gate:** screen the model's proposed tool calls (destructive arguments, argument provenance in tainted sessions).
7. **Kill-switch** on text + tool-call fingerprint.
8. **Egress:** credentials redacted. In tainted sessions, external images and model-composed links are removed (exfiltration channels).
9. **Audit log** (redacted), then the response is returned as JSON or replayed as SSE when `stream=true`.

## Deployment Notes
- Single worker: rate limits, loop history and the token vault are in-process memory.
- Default storage is SQLite at `Agent_Firewall/agent_firewall.db`; set `DATABASE_URL` for Postgres.
