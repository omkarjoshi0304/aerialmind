"""Tests for the internal publish-subscribe message bus.

Verifies:
1. Subscribe/publish delivers messages to correct subscribers
2. Unsubscribe stops delivery
3. Subscriber exceptions are isolated and logged
4. Last-value retention works for late-joining subscribers
5. Real frozen dataclass payloads round-trip correctly
"""

from __future__ import annotations

import logging

import pytest

from aerialmind.core.bus import MessageBus
from aerialmind.core.types import LinkStatus, NavState

# ---------------------------------------------------------------------------
# Subscribe / Publish
# ---------------------------------------------------------------------------


class TestSubscribePublish:
    @pytest.fixture()
    def bus(self) -> MessageBus:
        return MessageBus()

    def test_subscriber_receives_published_message(self, bus: MessageBus) -> None:
        received: list[object] = []
        bus.subscribe("test", received.append)
        bus.publish("test", "hello")
        assert received == ["hello"]

    def test_multiple_subscribers_all_receive(self, bus: MessageBus) -> None:
        results_a: list[object] = []
        results_b: list[object] = []
        bus.subscribe("test", results_a.append)
        bus.subscribe("test", results_b.append)
        bus.publish("test", 42)
        assert results_a == [42]
        assert results_b == [42]

    def test_subscriber_only_receives_subscribed_topic(self, bus: MessageBus) -> None:
        received: list[object] = []
        bus.subscribe("topicA", received.append)
        bus.publish("topicB", "wrong topic")
        assert received == []

    def test_publish_with_no_subscribers_does_not_raise(self, bus: MessageBus) -> None:
        bus.publish("nobody_listening", "payload")

    def test_duplicate_subscription_receives_twice(self, bus: MessageBus) -> None:
        received: list[object] = []
        bus.subscribe("test", received.append)
        bus.subscribe("test", received.append)
        bus.publish("test", "dup")
        assert received == ["dup", "dup"]

    def test_publish_frozen_dataclass_payload(self, bus: MessageBus) -> None:
        received: list[object] = []
        bus.subscribe("nav", received.append)
        nav = NavState(
            position=(37.7749, -122.4194, 100.0),
            velocity=(0.0, 0.0, 0.0),
            attitude_wxyz=(1.0, 0.0, 0.0, 0.0),
            position_uncertainty=(1.0, 1.0, 2.0),
            coordinate_frame="WGS84",
            mono_ts=1.0,
        )
        bus.publish("nav", nav)
        assert received == [nav]
        assert received[0] is nav


# ---------------------------------------------------------------------------
# Unsubscribe
# ---------------------------------------------------------------------------


class TestUnsubscribe:
    @pytest.fixture()
    def bus(self) -> MessageBus:
        return MessageBus()

    def test_unsubscribed_callback_stops_receiving(self, bus: MessageBus) -> None:
        received: list[object] = []
        bus.subscribe("test", received.append)
        bus.unsubscribe("test", received.append)
        bus.publish("test", "after unsub")
        assert received == []

    def test_unsubscribe_nonexistent_raises_value_error(self, bus: MessageBus) -> None:
        with pytest.raises(ValueError):
            bus.unsubscribe("test", lambda x: None)

    def test_unsubscribe_one_of_many(self, bus: MessageBus) -> None:
        results_a: list[object] = []
        results_b: list[object] = []
        bus.subscribe("test", results_a.append)
        bus.subscribe("test", results_b.append)
        bus.unsubscribe("test", results_a.append)
        bus.publish("test", "only b")
        assert results_a == []
        assert results_b == ["only b"]


# ---------------------------------------------------------------------------
# Exception isolation
# ---------------------------------------------------------------------------


class TestExceptionIsolation:
    @pytest.fixture()
    def bus(self) -> MessageBus:
        return MessageBus()

    def test_exception_in_subscriber_does_not_stop_others(
        self, bus: MessageBus,
    ) -> None:
        results: list[object] = []

        def bad_callback(_payload: object) -> None:
            msg = "subscriber crash"
            raise RuntimeError(msg)

        bus.subscribe("test", bad_callback)
        bus.subscribe("test", results.append)
        bus.publish("test", "payload")
        assert results == ["payload"]

    def test_exception_is_logged(
        self, bus: MessageBus, caplog: pytest.LogCaptureFixture,
    ) -> None:
        def bad_callback(_payload: object) -> None:
            msg = "boom"
            raise RuntimeError(msg)

        bus.subscribe("test", bad_callback)
        with caplog.at_level(logging.ERROR):
            bus.publish("test", "payload")
        assert "boom" in caplog.text

    def test_publish_continues_after_exception(self, bus: MessageBus) -> None:
        call_order: list[str] = []

        def first(_: object) -> None:
            call_order.append("first")

        def crasher(_: object) -> None:
            call_order.append("crasher")
            msg = "crash"
            raise RuntimeError(msg)

        def third(_: object) -> None:
            call_order.append("third")

        bus.subscribe("test", first)
        bus.subscribe("test", crasher)
        bus.subscribe("test", third)
        bus.publish("test", "go")
        assert call_order == ["first", "crasher", "third"]


# ---------------------------------------------------------------------------
# Last-value retention
# ---------------------------------------------------------------------------


class TestGetLast:
    @pytest.fixture()
    def bus(self) -> MessageBus:
        return MessageBus()

    def test_get_last_returns_none_before_any_publish(self, bus: MessageBus) -> None:
        assert bus.get_last("nonexistent") is None

    def test_get_last_returns_most_recent_payload(self, bus: MessageBus) -> None:
        bus.publish("test", "first")
        bus.publish("test", "second")
        assert bus.get_last("test") == "second"

    def test_get_last_is_per_topic(self, bus: MessageBus) -> None:
        bus.publish("a", 1)
        bus.publish("b", 2)
        assert bus.get_last("a") == 1
        assert bus.get_last("b") == 2


# ---------------------------------------------------------------------------
# has_subscribers
# ---------------------------------------------------------------------------


class TestHasSubscribers:
    @pytest.fixture()
    def bus(self) -> MessageBus:
        return MessageBus()

    def test_no_subscribers_initially(self, bus: MessageBus) -> None:
        assert bus.has_subscribers("test") is False

    def test_has_subscribers_after_subscribe(self, bus: MessageBus) -> None:
        bus.subscribe("test", lambda x: None)
        assert bus.has_subscribers("test") is True

    def test_no_subscribers_after_unsubscribe_all(self, bus: MessageBus) -> None:
        callback = lambda x: None  # noqa: E731
        bus.subscribe("test", callback)
        bus.unsubscribe("test", callback)
        assert bus.has_subscribers("test") is False


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------


class TestClear:
    @pytest.fixture()
    def bus(self) -> MessageBus:
        return MessageBus()

    def test_clear_removes_all_subscriptions(self, bus: MessageBus) -> None:
        bus.subscribe("test", lambda x: None)
        bus.clear()
        assert bus.has_subscribers("test") is False

    def test_clear_removes_all_last_values(self, bus: MessageBus) -> None:
        bus.publish("test", "value")
        bus.clear()
        assert bus.get_last("test") is None


# ---------------------------------------------------------------------------
# Integration with real payloads
# ---------------------------------------------------------------------------


class TestWithRealPayloads:
    @pytest.fixture()
    def bus(self) -> MessageBus:
        return MessageBus()

    def test_nav_state_roundtrip(self, bus: MessageBus) -> None:
        received: list[object] = []
        bus.subscribe("nav.state", received.append)
        nav = NavState(
            position=(37.7749, -122.4194, 100.0),
            velocity=(1.0, 0.0, -0.5),
            attitude_wxyz=(1.0, 0.0, 0.0, 0.0),
            position_uncertainty=(2.0, 2.0, 5.0),
            coordinate_frame="WGS84",
            mono_ts=1234.567,
        )
        bus.publish("nav.state", nav)
        assert len(received) == 1
        assert received[0] is nav
        assert bus.get_last("nav.state") is nav

    def test_link_status_roundtrip(self, bus: MessageBus) -> None:
        received: list[object] = []
        bus.subscribe("comms.link_status", received.append)
        link = LinkStatus(
            connected=True,
            latency_ms=45.0,
            rssi_dbm=-65.0,
            quality_pct=92.0,
            last_heartbeat_ts=1234.0,
        )
        bus.publish("comms.link_status", link)
        assert len(received) == 1
        assert received[0] == link
