"""Compatibility re-exports for agent classes.

The canonical assignment-required agent implementations live in the top-level
`agents/` folder (e.g. `agents/ceo_agent.py`). This module exists so other
code can import agents via `multi_agent_system.agents` without needing a
subpackage directory.
"""

from agents.ceo_agent import CEOAgent
from agents.engineer_agent import EngineerAgent
from agents.marketing_agent import MarketingAgent
from agents.product_agent import ProductAgent
from agents.qa_agent import QAAgent

__all__ = ["CEOAgent", "ProductAgent", "EngineerAgent", "MarketingAgent", "QAAgent"]

