# Agent Firewall

A security proxy that sits between AI agents and LLM providers. Agents talk to it
exactly like the OpenAI API; the firewall inspects everything going in and out.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the design and
[docs/PROMPT_INJECTION.md](docs/PROMPT_INJECTION.md) for the injection defenses.

## Setup

```bash
pip install -r Agent_Firewall/requirements.txt
# optional, for the ML injection classifier (≈1 GB download on first use):
pip install -r prompt_injection/requirements.txt

cp .env.example Agent_Firewall/.env   # then fill in OPENROUTER_KEY etc.
```

## Run

```bash
# from the repository root
uvicorn Agent_Firewall.main:app --port 8000 --workers 1
```

Open http://localhost:8000. The dashboard asks for the admin token: set
`FIREWALL_ADMIN_TOKEN` in `.env`, or copy the one printed in the server log.

Create a virtual key in the **API Keys** tab (shown once; only its hash is stored), then:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/api/v1", api_key="agf-live-...")
client.chat.completions.create(model="openrouter/openai/gpt-4o-mini",
                               messages=[{"role": "user", "content": "Hello"}])
```

Tool calling and `stream=True` work as with the OpenAI API.

## Test

```bash
pytest tests
```

## Known limitations

- Run a **single worker**: rate limits, kill-switch history and the PII token vault are
  kept in process memory and are not shared between workers or kept across restarts.
- Streaming is emulated: the full reply is inspected first, then sent as one SSE chunk.
- Detection is probabilistic; tool-call gating and output filtering are what limit damage
  when an injection gets through.
- Credentials that were committed to this repository's history must be rotated with the
  provider; removing them from the code does not revoke them.
