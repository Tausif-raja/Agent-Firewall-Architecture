# Agent Firewall Architecture

## Overview
The Agent Firewall acts as a robust middleware layer positioned between the user and the LLM (Large Language Model) or AI Agents. Its primary purpose is to inspect, filter, and secure interactions, mitigating risks associated with prompt injections, data leakage, and runaway agent loops.

## Core Components

1. **Prompt Injection Detection & Prevention**
   - Intercepts incoming user prompts before they reach the LLM.
   - Evaluates prompts for malicious intent, jailbreak attempts, or unauthorized command injections.
   - *(Note: Specific technologies or models to be determined—e.g., heuristic filters, secondary LLM evaluation, or specialized classifier models).*

2. **Agent Loop Management (Infinite Loop Prevention)**
   - Monitors the reasoning and action loops of AI agents.
   - Compares successive reasoning tokens or states to determine if actual progress is being made.
   - Uses **Embeddings** to compare reasoning tokens from one step to the next. If the semantic similarity is too high over multiple steps (indicating repetitive thought processes without progression), the firewall intercepts and halts the unnecessary looping.

3. **Private Data (PII) Detection & Redaction**
   - Scans both incoming prompts and outgoing LLM responses for sensitive personal data.
   - Utilizes tools like **Microsoft PII detection (e.g., Microsoft Presidio)** to identify and mask/redact sensitive information.
   - Protects against accidental data leaks and exfiltration vulnerabilities.

4. **Vulnerability Mitigation**
   - Acts as a general shield against various known vulnerabilities in AI agents.
   - Ensures that the agent stays within its defined operational boundaries.

## Architecture Flow (Middle-ware Instance)
1. **User Input:** User submits a prompt.
2. **Ingress Filtering (Firewall):**
   - PII Detection scans and masks private data.
   - Prompt Injection Detection analyzes the prompt for malicious instructions.
   - If a threat is detected, the request is blocked or sanitized.
3. **LLM/Agent Processing:** The safe prompt is passed to the LLM/Agent.
4. **Agent Monitoring (Firewall):**
   - As the agent reasons (Chain of Thought), the firewall embeds and compares consecutive reasoning outputs.
   - Halts the agent if an infinite or unproductive loop is detected.
5. **Egress Filtering (Firewall):**
   - The final response is scanned again for PII or other sensitive content before being released.
6. **User Output:** The safe, filtered response is delivered to the user.
