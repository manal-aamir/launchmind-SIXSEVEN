# InvoiceHound — Multi-Agent System
### FAST NUCES — Agentic AI Assignment 2

> A fully autonomous 5-agent pipeline that builds, launches, and manages a
> freelance invoicing startup — from idea decomposition to GitHub PR, marketing
> launch, and escalating payment reminders.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure .env (see section below)
cp .env.example .env   # then fill in your keys

# 3. Run the Flask web UI
python3 app.py
# → open http://127.0.0.1:8080

# 4. Run the full 5-agent pipeline directly
python3 run_multi_agent_demo.py

# 5. Run the AutoGen demo
python3 run_autogen_demo.py
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       Flask Web UI  (:8080)                     │
│   Create Invoice → Trigger Agents → View Messages → Download   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   CEO Agent     │  ← Groq LLaMA-3.3-70b
                    │  (Orchestrator) │
                    └──┬──┬──┬──┬────┘
                       │  │  │  │
          ┌────────────┘  │  │  └──────────────────────────┐
          │               │  │                             │
   ┌──────▼──────┐  ┌─────▼──────┐  ┌──────────────┐  ┌───▼──────┐
   │ Product     │  │ Engineer   │  │ Marketing    │  │ QA       │
   │ Agent       │  │ Agent      │  │ Agent        │  │ Agent    │
   │ OpenAI      │  │ OpenAI     │  │ OpenAI       │  │ Groq     │
   │ gpt-4o-mini │  │ gpt-4o-mini│  │ gpt-4o-mini  │  │ LLaMA-3.3│
   └─────────────┘  └────────────┘  └──────────────┘  └──────────┘
                           │                │
                    ┌──────▼──────┐  ┌──────▼──────┐
                    │  GitHub API │  │ SendGrid    │
                    │  Branch/PR  │  │ Slack       │
                    └─────────────┘  └─────────────┘
```

### Multi-LLM Integration (Bonus +2%)
| Agent | LLM Provider | Model | Role |
|-------|-------------|-------|------|
| CEO | **Groq** | llama-3.3-70b-versatile | Decompose, review, summarise |
| Product | **OpenAI** | gpt-4o-mini | Product spec generation |
| Engineer | **OpenAI** | gpt-4o-mini | HTML landing page generation |
| Marketing | **OpenAI** | gpt-4o-mini | Copy, email, social posts |
| QA | **Groq** | llama-3.3-70b-versatile | HTML + copy review |

---

## Agent Roles

### Agent 1: CEO (Orchestrator)
- Receives startup idea from Flask UI or `startup_config.json`
- Uses **Groq LLaMA-3.3-70b** to decompose it into structured tasks
- Sends JSON task messages to Product, Engineer, Marketing via `MessageBus`
- Reviews each output and sends **revision_request** if score < threshold
- Runs **up to 2 revision cycles** per agent (multiple feedback loops)
- After QA, posts final summary to Slack `#launches`
- Gracefully handles failures: retries 3× with backoff, logs to `AgentFailure`

### Agent 2: Product
- Generates full product spec: value proposition, 3 personas, 5 ranked features, 3 user stories
- Explicitly covers **Day 1/Day 7/Day 14 reminder escalation** and **hour-based split logic**
- LLM: OpenAI gpt-4o-mini

### Agent 3: Engineer
- Generates complete `index.html` landing page from product spec
- Creates GitHub issue → branch `agent-landing-page` → commits `index.html` → opens PR
- LLM: OpenAI gpt-4o-mini · API: GitHub REST

### Agent 4: Marketing
- Generates tagline, landing-page description, cold outreach email, 3 social posts
- Sends cold email via **SendGrid** to test inbox
- Posts **Slack Block Kit** launch message to `#launches` with PR link
- LLM: OpenAI gpt-4o-mini · APIs: SendGrid, Slack

### Agent 5: QA / Reviewer
- Reviews Engineer's HTML and Marketing's copy against product spec
- Posts **≥ 2 inline PR review comments** on GitHub via REST API
- Returns structured pass/fail report to CEO
- LLM: **Groq LLaMA-3.3-70b** (different provider = multi-LLM bonus)

---

## InvoiceHound Business Logic

### Invoice Generation
- Professional HTML invoice with line items, GST (10%), and grand total
- Internal payment split section showing each member's share by hours
- Auto-sends initial invoice email (LLM-written, not hardcoded) via SendGrid

### Escalating Reminders
| Day | Action | Channel |
|-----|--------|---------|
| Day 1 | Polite Slack Block Kit message to team | `#launches` |
| Day 7 | Firmer Slack nudge to team | `#launches` |
| Day 14 | **Formal final notice email** to client with full HTML invoice embedded | SendGrid |

### Payment Distribution
- When client pays, CEO splits payment proportionally by hours worked
- Each team member notified via Slack with their individual amount

---

## Message Schema (Section 4.1)

Every inter-agent message follows this exact structure:

```json
{
  "message_id":        "msg-a1b2c3d4",
  "from_agent":        "ceo",
  "to_agent":          "product",
  "message_type":      "task",
  "payload": {
    "idea":       "InvoiceHound — freelance invoice and payment-split tool",
    "brief":      "Define user personas and top 5 features",
    "expected":   ["Value proposition", "Three personas", "Five features"],
    "constraints": ["Must cover Day 1/7/14 reminders", "Must cover hour splits"]
  },
  "timestamp":         "2026-03-25T09:00:00Z",
  "parent_message_id": null
}
```

`message_type` values: `task` → `result` → `revision_request` → `confirmation`

### Viewing Message History
Open **http://127.0.0.1:8080/messages** — shows every message the CEO sent
and received, colour-coded by type, with expandable payloads.

To answer the evaluator question *"Show me every message the CEO sent and received"*:
```bash
# All CEO messages from most recent pipeline run
cat logs/message_log_*.json | python3 -m json.tool | grep -A 20 '"from_agent": "ceo"'
```

---

## Bonus Features

| Bonus | Status | Implementation |
|-------|--------|----------------|
| QA agent (+5%) | ✅ | `multi_agent_system/agents/qa.py` — Groq reviews HTML+copy, posts PR comments |
| Redis pub/sub (+3%) | ✅ | `multi_agent_system/redis_bus.py` — falls back to in-memory if Redis unavailable |
| Graceful failure handling (+3%) | ✅ | `multi_agent_system/retry.py` — `safe_call()` with exponential backoff, `AgentFailure` reported to CEO |
| Multiple revision cycles (+2%) | ✅ | CEO sends up to 2 `revision_request` messages per agent (`max_revisions=2`) |
| Different LLM providers (+2%) | ✅ | CEO+QA = Groq LLaMA-3.3-70b · Product+Engineer+Marketing = OpenAI gpt-4o-mini |

---

## .env Configuration

```bash
# GitHub
GITHUB_TOKEN='ghp_...'
GITHUB_REPO='username/repo-name'

# Slack
SLACK_BOT_TOKEN='xoxb-...'
SLACK_CHANNEL_ID='C...'
LAUNCHES_CHANNEL_ID='C...'

# SendGrid
SENDGRID_API_KEY='SG...'
SENDGRID_FROM_EMAIL='verified@yourdomain.com'
SENDGRID_TO_EMAIL='test@yourdomain.com'

# OpenAI (Product, Engineer, Marketing agents)
OPENAI_API_KEY='sk-...'
OPENAI_MODEL='gpt-4o-mini'

# Groq (CEO + QA agents)
GROQ_API_KEY='gsk_...'
GROQ_MODEL='llama-3.3-70b-versatile'

# Redis (optional — for pub/sub bonus)
REDIS_HOST='localhost'
REDIS_PORT='6379'
```

---

## Repository Structure

```
Assignment 2/
├── app.py                          # Flask web UI (port 8080)
├── run_multi_agent_demo.py         # Direct 5-agent pipeline runner
├── run_autogen_demo.py             # AutoGen (Microsoft) agent demo
├── invoicehound_demo.py            # Standalone invoice/reminder demo
├── startup_config.json             # Pipeline configuration
├── requirements.txt
├── README_agents.md                # ← this file
│
├── multi_agent_system/
│   ├── agents/
│   │   ├── ceo.py                  # Orchestrator — Groq LLM
│   │   ├── product.py              # Product spec — OpenAI
│   │   ├── engineer.py             # Landing page + GitHub — OpenAI
│   │   ├── marketing.py            # Copy + SendGrid + Slack — OpenAI
│   │   └── qa.py                   # Review + PR comments — Groq
│   ├── integrations/
│   │   ├── github_client.py        # GitHub REST API
│   │   ├── sendgrid_client.py      # SendGrid email
│   │   └── slack_client.py         # Slack Block Kit
│   ├── models.py                   # Message schema + MessageBus
│   ├── redis_bus.py                # Redis pub/sub transport (bonus)
│   ├── retry.py                    # Graceful failure + retry (bonus)
│   ├── llm_client.py               # OpenAI client
│   ├── groq_client.py              # Groq client (CEO + QA)
│   ├── invoice_engine.py           # Invoice HTML + payment splits
│   └── reminder_engine.py          # Day 1/7/14 reminder logic
│
├── templates/
│   ├── base.html
│   ├── new_invoice.html            # Create invoice form
│   ├── dashboard.html              # Invoice list + actions
│   └── messages.html              # Message history viewer
│
└── logs/
    ├── ceo_decisions_*.json        # CEO decision log per run
    └── message_log_*.json          # Full inter-agent message log
```

---

## How to Demo

### Full Pipeline Demo
```bash
python3 run_multi_agent_demo.py
```
Watch the CEO decompose the idea, send tasks to all agents, review outputs,
and post to Slack. Check `logs/` for JSON message logs.

### Flask UI Demo
```bash
python3 app.py
```
1. Open http://127.0.0.1:8080
2. Fill in client name, email, project, invoice items, and team member hours
3. Click **"Generate Invoice & Launch Agents"**
4. Check email inbox for the invoice email (SendGrid)
5. Check Slack `#launches` for the launch announcement
6. Open **Messages** tab to see full CEO message history
7. On the Dashboard, trigger Day 1 / Day 7 / Day 14 reminders
8. Click **Mark as Paid** to distribute splits and notify team on Slack
9. Download individual team member earnings statements

### AutoGen Demo
```bash
python3 run_autogen_demo.py
```
Microsoft AutoGen orchestrates all 5 agents in a group chat with registered tools.

### Evaluator Queries
| Question | Answer |
|----------|--------|
| "Show me every message the CEO sent and received" | Open `/messages` in browser, or `cat logs/message_log_*.json` |
| "Where is the GitHub PR?" | Logged in `logs/ceo_decisions_*.json` under `engineer` stage |
| "Did the CEO send a revision request?" | Check Messages tab — filter `revision_request` type |
| "Which LLMs are used?" | CEO+QA: Groq LLaMA-3.3-70b · Others: OpenAI gpt-4o-mini |
| "What happens if an API call fails?" | `retry.py` retries 3× with backoff; `AgentFailure` logged to CEO |
