"""
Redis pub/sub message bus — Option C (bonus +3%).

Each agent subscribes to a Redis channel named after itself:
  agent:ceo, agent:product, agent:engineer, agent:marketing, agent:qa

The CEO publishes to the recipient's channel; agents listen on their own.
Falls back silently to the in-memory MessageBus when Redis is unavailable.

Usage:
    bus = RedisBus(host="localhost", port=6379)
    bus.publish("product", message_dict)          # CEO → product channel
    for msg in bus.listen("product", timeout=10): # product agent consuming
        handle(msg)
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable, Dict, Generator, List, Optional

try:
    import redis                          # type: ignore
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False


def _channel(agent_name: str) -> str:
    return f"agent:{agent_name}"


# ---------------------------------------------------------------------------
# RedisBus
# ---------------------------------------------------------------------------

class RedisBus:
    """
    Pub/sub transport layer.  Agents publish to recipient channels and
    subscribe to their own.  Full message history is kept in a Redis list
    `message_history` for the evaluator query.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        fallback_bus=None,      # in-memory MessageBus fallback
    ) -> None:
        self._fallback = fallback_bus
        self._available = False
        self._client: Any = None
        self._subscribers: Dict[str, Any] = {}
        self._history: List[Dict[str, Any]] = []   # local mirror for display
        self._lock = threading.Lock()

        if not _REDIS_AVAILABLE:
            print("[RedisBus] redis-py not installed — falling back to in-memory bus.")
            return

        try:
            self._client = redis.Redis(
                host=host, port=port, db=db,
                socket_connect_timeout=2,
                decode_responses=True,
            )
            self._client.ping()
            self._available = True
            print(f"[RedisBus] ✅ Connected to Redis at {host}:{port}")
        except Exception as exc:
            print(f"[RedisBus] ⚠  Redis unavailable ({exc}) — falling back to in-memory bus.")

    @property
    def is_redis(self) -> bool:
        return self._available

    # ------------------------------------------------------------------
    # Core pub/sub
    # ------------------------------------------------------------------

    def publish(self, to_agent: str, message: Dict[str, Any]) -> None:
        """Publish a message to an agent's channel."""
        with self._lock:
            self._history.append(message)

        if self._available:
            payload = json.dumps(message, ensure_ascii=False)
            self._client.publish(_channel(to_agent), payload)
            # persist to Redis list for history queries
            self._client.rpush("invoicehound:message_history", payload)
        elif self._fallback:
            self._fallback._messages.append(message)

    def subscribe(self, agent_name: str):
        """Return a Redis PubSub handle subscribed to this agent's channel."""
        if not self._available:
            return None
        ps = self._client.pubsub()
        ps.subscribe(_channel(agent_name))
        return ps

    def listen(
        self, agent_name: str, timeout: float = 30.0
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Blocking generator: yields deserialized messages sent to *agent_name*.
        Stops after *timeout* seconds without a new message.
        Falls back to an empty generator when Redis is unavailable.
        """
        ps = self.subscribe(agent_name)
        if ps is None:
            return

        deadline = time.monotonic() + timeout
        for raw_msg in ps.listen():
            if raw_msg["type"] == "message":
                yield json.loads(raw_msg["data"])
                deadline = time.monotonic() + timeout
            if time.monotonic() > deadline:
                break
        ps.unsubscribe()

    # ------------------------------------------------------------------
    # History queries (evaluator: "show every CEO message")
    # ------------------------------------------------------------------

    def all_messages(self) -> List[Dict[str, Any]]:
        if self._available:
            raw = self._client.lrange("invoicehound:message_history", 0, -1)
            return [json.loads(r) for r in raw]
        return list(self._history)

    def ceo_messages(self) -> List[Dict[str, Any]]:
        return [
            m for m in self.all_messages()
            if m.get("from_agent") == "ceo" or m.get("to_agent") == "ceo"
        ]

    def flush(self) -> None:
        """Clear the history (for testing)."""
        if self._available:
            self._client.delete("invoicehound:message_history")
        with self._lock:
            self._history.clear()
