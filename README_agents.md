# Multi-Agent Startup System

This implementation includes 5 agents:

1. `CEOAgent` (orchestrator)
2. `ProductAgent`
3. `EngineerAgent`
4. `MarketingAgent`
5. `QAAgent`

## What the CEO does

- Receives a startup idea string.
- Uses an LLM to decompose that idea into structured JSON tasks for Product, Engineer, and Marketing.
- Sends each task to the matching agent.
- Uses an LLM review step per output to accept/reject quality.
- Sends revision instructions when output is weak (up to 2 rounds).
- Runs QA cross-check.
- Posts final summary to Slack channel.
- Saves a decision log in `logs/`.

## Required `.env` keys

- `OPENAI_API_KEY` (for real LLM calls)
- `OPENAI_MODEL` (optional, default `gpt-4o-mini`)
- `GITHUB_TOKEN`
- `GITHUB_REPO` (e.g. `manal-aamir/launchmind-SIXSEVEN`)
- `SLACK_BOT_TOKEN`
- `SLACK_CHANNEL_ID` (team channel where final summary is posted)
- `LAUNCHES_CHANNEL_ID` (channel for marketing launch message, e.g. `#launches`)
- `SENDGRID_API_KEY`
- `SENDGRID_FROM_EMAIL`
- `SENDGRID_TO_EMAIL`

Existing keys used by your earlier tasks can stay in `.env`.

## Run

```bash
cd "/Users/4star/Documents/8th semster/Agentic AI/Assignment 2"
python3 -m pip install -r requirements.txt
python3 run_multi_agent_demo.py --idea "Your startup idea here" --execute-actions
```

Use dry-run (no Slack post):

```bash
python3 run_multi_agent_demo.py --dry-run
```

