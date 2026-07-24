"""AerialMind core data types.

All enums, value objects, and data structures shared across modules.
Every module imports from here — this file has ZERO internal dependencies.
"""

from __future__ import annotations

from enum import Enum, auto


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class OperatingMode(Enum):
    """Drone operating mode — determines detection models, ROE policy, and threat weights."""

    MILITARY = auto()
    CIVIL = auto()


class ThreatLevel(Enum):
    """Composite threat score mapped to discrete levels.

    Explicit integer values allow numeric comparison:
        if assessment.level.value >= ThreatLevel.HIGH.value: ...
    """

    NONE = 0
    LOW = 1
    MODERATE = 2
    HIGH = 3
    CRITICAL = 4


class CERState(Enum):
    """Cognitive Edge Resilience state machine.

    Tracks GPS health and link health on two independent axes.
    Only when BOTH are lost (CER_FULL) does the drone enter fully
    autonomous survival behavior.
    """

    NOMINAL = auto()
    GPS_DEGRADED = auto()
    GPS_DENIED = auto()
    LINK_DEGRADED = auto()
    LINK_LOST = auto()
    CER_PARTIAL = auto()
    CER_FULL = auto()
    SAFE_ZONE_RETURN = auto()
    EMERGENCY_LAND = auto()
    LANDED = auto()


class DecisionState(Enum):
    """Decision engine state machine — what the drone is currently doing."""

    IDLE = auto()
    MONITORING = auto()
    TRACKING = auto()
    ALERTING = auto()
    AUTONOMOUS_ACTION = auto()
    RETURNING = auto()
    LANDED = auto()


class ResourcePriority(Enum):
    """QoS priority for compute resource allocation.

    The Orchestrator's QoS Manager uses these to dynamically throttle
    modules when compute is scarce. In CER mode, navigation gets
    CRITICAL priority and vision drops to LOW/BACKGROUND.
    """

    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4
