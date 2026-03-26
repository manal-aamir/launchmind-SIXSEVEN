# InvoiceHound — Multi-Agent System (FAST NUCES Agentic AI Assignment 2)

InvoiceHound is a tool for freelance teams that generates **one professional client invoice**, splits earnings **internally by hours logged**, and automatically escalates payment reminders on **Day 1 / Day 7 / Day 14** until the client pays.

This repository contains a 5-agent Multi‑Agent System (MAS) that runs end‑to‑end: the CEO decomposes tasks, Product writes the product spec, Engineer generates a landing page and opens a GitHub PR, Marketing prepares launch assets and posts to Slack, and QA reviews and comments on the PR.

---
<img width="1395" height="491" alt="Screenshot 2026-03-26 at 3 36 45 PM" src="https://github.com/user-attachments/assets/bc128ed6-23f1-4575-a28b-e83ea4065503" />


## Agent architecture (who talks to who)

- **CEO → Product**: task message (product spec JSON)
- **CEO → Engineer**: task message (landing page + GitHub issue/PR)
- **CEO → Marketing**: task message (tagline, email, social posts, Slack launch)
- **QA → GitHub**: PR review comments on `index.html`
- **All agents ↔ MessageBus**: every message is logged for `/messages` + evaluator inspection

---

## Setup

```bash
git clone https://github.com/manal-aamir/launchmind-SIXSEVEN.git
cd launchmind-SIXSEVEN

pip install -r requirements.txt

cp .env.example .env
# edit .env and add your real keys
```

### Run the system (end-to-end)

```bash
python3 main.py
```

### Run the Flask UI

```bash
python3 app.py
# open http://127.0.0.1:5000
```

---

## Platforms / integrations (what the agents do)

- **Groq**: primary LLM provider for agent generation and review
- **GitHub**: Engineer opens an issue + branch + commit + PR; QA posts PR review comments
- **Slack**: reminders + launch post + CEO final summary
- **SendGrid**: sends invoice emails, Day 14 reminder (PDF attachment), and payout PDFs to team members after payment
- **Redis (bonus)**: pub/sub message transport (falls back to in-memory if Redis not running)

---

## Slack workspace link / screenshots

Add your Slack invite link here (or include screenshots of the bot posting in `#launches`):
- Slack invite: *https://app.slack.com/client/T71NF2AUX/C0ANVB4HEE8*

---

## Engineer agent PR link

Engineer agent creates a landing-page PR. Example from a real run:
- PR: `https://github.com/manal-aamir/launchmind-SIXSEVEN/pull/2`

---

## Repository structure (grader checklist)

```
.
├── agents/
│   ├── ceo_agent.py
│   ├── product_agent.py
│   ├── engineer_agent.py
│   ├── marketing_agent.py
│   └── qa_agent.py
├── main.py
├── message_bus.py
├── requirements.txt
├── .env.example
├── .gitignore
└── multi_agent_system/            # full implementation (wrappers above call into this)
```
