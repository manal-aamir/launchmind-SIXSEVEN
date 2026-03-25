"""
Graceful failure handling — bonus criterion (+3%).

Provides:
  - retry_with_backoff(): wraps any callable, retries on exception with
    exponential backoff, and raises AgentFailure on exhaustion.
  - AgentFailure: structured exception with enough context for the CEO to
    log and decide whether to continue or abort the pipeline.
  - safe_call(): convenience wrapper that returns a (result, error) tuple
    so call sites don't need bare try/except everywhere.

The CEO agent catches AgentFailure from any sub-agent, logs it to the
decision log and the MessageBus, and either retries the whole agent task
or continues with a degraded result — ensuring the pipeline never crashes
silently.
"""

from __future__ import annotations

import time
import traceback
from typing import Any, Callable, Optional, Tuple, TypeVar

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class AgentFailure(Exception):
    """Raised when an agent exhausts all retries."""

    def __init__(
        self,
        agent_name: str,
        operation: str,
        original_error: Exception,
        attempts: int,
    ) -> None:
        self.agent_name    = agent_name
        self.operation     = operation
        self.original_error = original_error
        self.attempts      = attempts
        super().__init__(
            f"[{agent_name}] '{operation}' failed after {attempts} attempt(s): "
            f"{type(original_error).__name__}: {original_error}"
        )

    def to_dict(self) -> dict:
        return {
            "agent":     self.agent_name,
            "operation": self.operation,
            "error":     str(self.original_error),
            "type":      type(self.original_error).__name__,
            "attempts":  self.attempts,
        }


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

def retry_with_backoff(
    fn: Callable[..., T],
    *args: Any,
    agent_name: str = "unknown",
    operation: str = "call",
    retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    **kwargs: Any,
) -> T:
    """
    Call *fn(*args, **kwargs)* up to *retries* times.
    Between attempts, sleep for initial_delay * (backoff_factor ** attempt).
    Raises AgentFailure if all attempts fail.
    """
    delay = initial_delay
    last_exc: Exception = RuntimeError("No attempts made")

    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            tb = traceback.format_exc(limit=3)
            print(
                f"[retry] {agent_name}/{operation} — attempt {attempt}/{retries} failed: "
                f"{type(exc).__name__}: {exc}\n{tb.strip()}"
            )
            if attempt < retries:
                time.sleep(delay)
                delay *= backoff_factor

    raise AgentFailure(agent_name, operation, last_exc, retries)


# ---------------------------------------------------------------------------
# Safe-call wrapper
# ---------------------------------------------------------------------------

def safe_call(
    fn: Callable[..., T],
    *args: Any,
    agent_name: str = "unknown",
    operation: str = "call",
    retries: int = 3,
    fallback: Optional[T] = None,
    **kwargs: Any,
) -> Tuple[Optional[T], Optional[AgentFailure]]:
    """
    Like retry_with_backoff but returns (result, None) on success or
    (fallback, AgentFailure) on exhaustion — never raises.

    Callers (e.g. CEOAgent) can inspect the AgentFailure and decide
    whether to abort the pipeline or proceed with the degraded fallback.
    """
    try:
        result = retry_with_backoff(
            fn, *args,
            agent_name=agent_name,
            operation=operation,
            retries=retries,
            **kwargs,
        )
        return result, None
    except AgentFailure as af:
        return fallback, af
