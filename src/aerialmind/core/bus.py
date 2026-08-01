"""AerialMind internal publish-subscribe message bus.

All inter-module communication flows through this bus. A module publishes
an event on a named topic and every subscriber receives it — no direct
coupling between publisher and subscriber.

Not thread-safe for MVP. Thread safety will be added in Feature 9
(Orchestrator) when real threading is introduced.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class MessageBus:
    """In-process publish-subscribe event bus."""

    def __init__(self) -> None:
        self._subscribers: defaultdict[str, list[Callable[[Any], None]]] = defaultdict(
            list,
        )
        self._last_values: dict[str, object] = {}

    def subscribe(self, topic: str, callback: Callable[[Any], None]) -> None:
        """Register a callback for a topic. Duplicate subscriptions are allowed."""
        self._subscribers[topic].append(callback)

    def unsubscribe(self, topic: str, callback: Callable[[Any], None]) -> None:
        """Remove a specific callback from a topic.

        Raises ValueError if the callback is not registered on the topic.
        """
        self._subscribers[topic].remove(callback)

    def publish(self, topic: str, payload: object) -> None:
        """Publish a payload to all subscribers on a topic.

        Stores the payload as the last value for the topic.
        Catches and logs any exception raised by a subscriber callback,
        then continues dispatching to remaining subscribers.
        """
        self._last_values[topic] = payload
        for callback in list(self._subscribers[topic]):
            try:
                callback(payload)
            except Exception:
                logger.exception(
                    "Subscriber %r raised on topic %r",
                    callback,
                    topic,
                )

    def get_last(self, topic: str) -> object | None:
        """Return the last published payload for a topic, or None if never published."""
        return self._last_values.get(topic)

    def has_subscribers(self, topic: str) -> bool:
        """Return True if any callbacks are registered for the topic."""
        return len(self._subscribers[topic]) > 0

    def clear(self) -> None:
        """Remove all subscriptions and last values."""
        self._subscribers.clear()
        self._last_values.clear()
