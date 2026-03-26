"""
Assignment-required shared messaging implementation.

This file is a thin wrapper over the actual implementation in
`multi_agent_system/models.py` to match the required repo structure.
"""

from multi_agent_system.models import MessageBus, make_message

__all__ = ["MessageBus", "make_message"]

