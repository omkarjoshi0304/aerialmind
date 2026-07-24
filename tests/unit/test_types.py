"""Tests for core data types — enums, frozen dataclasses, and protocols.

Verifies:
1. Enum members exist and have correct values
2. Frozen dataclasses are truly immutable
3. Protocol structural typing works (isinstance checks)
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Optional

import numpy as np
import pytest

from aerialmind.core.types import (
    ActionDecision,
    BehaviorEvent,
    CERState,
    DecisionState,
    Detection,
    GPSFix,
    LinkStatus,
    NavState,
    OperatingMode,
    OpticalFlowReading,
    PoseResult,
    Recommendation,
    ResourcePriority,
    ThreatAssessment,
    ThreatLevel,
    TimestampedFrame,
    TimestampedIMU,
    Track,
)
from aerialmind.core.protocols import (
    AcceleratorHAL,
    AltimeterHAL,
    BehaviorAnalyzerInterface,
    CameraHAL,
    DecisionEngineInterface,
    GPSHAL,
    IMUHAL,
    NavigationEKFInterface,
    ObjectDetectorInterface,
    OpticalFlowHAL,
    PoseEstimatorInterface,
    TelemetryInterface,
    TrackerInterface,
    VIOEngineInterface,
)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestOperatingMode:
    def test_has_military_and_civil(self) -> None:
        assert OperatingMode.MILITARY is not None
        assert OperatingMode.CIVIL is not None

    def test_members_are_distinct(self) -> None:
        assert OperatingMode.MILITARY != OperatingMode.CIVIL

    def test_member_count(self) -> None:
        assert len(OperatingMode) == 2


class TestThreatLevel:
    def test_values_are_ordered(self) -> None:
        assert ThreatLevel.NONE.value < ThreatLevel.LOW.value
        assert ThreatLevel.LOW.value < ThreatLevel.MODERATE.value
        assert ThreatLevel.MODERATE.value < ThreatLevel.HIGH.value
        assert ThreatLevel.HIGH.value < ThreatLevel.CRITICAL.value

    def test_none_is_zero(self) -> None:
        assert ThreatLevel.NONE.value == 0

    def test_critical_is_four(self) -> None:
        assert ThreatLevel.CRITICAL.value == 4

    def test_member_count(self) -> None:
        assert len(ThreatLevel) == 5


class TestCERState:
    def test_has_all_states(self) -> None:
        expected = {
            "NOMINAL", "GPS_DEGRADED", "GPS_DENIED",
            "LINK_DEGRADED", "LINK_LOST", "CER_PARTIAL",
            "CER_FULL", "SAFE_ZONE_RETURN", "EMERGENCY_LAND", "LANDED",
        }
        actual = {s.name for s in CERState}
        assert actual == expected

    def test_member_count(self) -> None:
        assert len(CERState) == 10


class TestDecisionState:
    def test_has_all_states(self) -> None:
        expected = {
            "IDLE", "MONITORING", "TRACKING", "ALERTING",
            "AUTONOMOUS_ACTION", "RETURNING", "LANDED",
        }
        actual = {s.name for s in DecisionState}
        assert actual == expected

    def test_member_count(self) -> None:
        assert len(DecisionState) == 7


class TestResourcePriority:
    def test_critical_is_highest_priority(self) -> None:
        assert ResourcePriority.CRITICAL.value < ResourcePriority.HIGH.value

    def test_values_are_ordered(self) -> None:
        values = [p.value for p in ResourcePriority]
        assert values == sorted(values)

    def test_member_count(self) -> None:
        assert len(ResourcePriority) == 5


# ---------------------------------------------------------------------------
# Frozen dataclass tests — verify immutability
# ---------------------------------------------------------------------------


class TestTimestampedFrame:
    @pytest.fixture()
    def frame(self) -> TimestampedFrame:
        return TimestampedFrame(
            frame=np.zeros((720, 1280, 3), dtype=np.uint8),
            mono_ts=1.0,
            seq_id=1,
            width=1280,
            height=720,
        )

    def test_fields_accessible(self, frame: TimestampedFrame) -> None:
        assert frame.mono_ts == 1.0
        assert frame.seq_id == 1
        assert frame.width == 1280
        assert frame.height == 720
        assert frame.frame.shape == (720, 1280, 3)

    def test_frozen(self, frame: TimestampedFrame) -> None:
        with pytest.raises(FrozenInstanceError):
            frame.mono_ts = 2.0  # type: ignore[misc]


class TestTimestampedIMU:
    @pytest.fixture()
    def imu(self) -> TimestampedIMU:
        return TimestampedIMU(
            accel=(0.0, 0.0, -9.81),
            gyro=(0.0, 0.0, 0.0),
            mono_ts=1.0,
        )

    def test_fields_accessible(self, imu: TimestampedIMU) -> None:
        assert imu.accel == (0.0, 0.0, -9.81)
        assert imu.gyro == (0.0, 0.0, 0.0)
        assert imu.mono_ts == 1.0

    def test_frozen(self, imu: TimestampedIMU) -> None:
        with pytest.raises(FrozenInstanceError):
            imu.mono_ts = 2.0  # type: ignore[misc]


class TestGPSFix:
    @pytest.fixture()
    def gps(self) -> GPSFix:
        return GPSFix(
            latitude=37.7749,
            longitude=-122.4194,
            altitude_msl=10.0,
            hdop=1.2,
            fix_type=3,
            num_satellites=12,
            mono_ts=1.0,
            valid=True,
        )

    def test_fields_accessible(self, gps: GPSFix) -> None:
        assert gps.latitude == pytest.approx(37.7749)
        assert gps.fix_type == 3
        assert gps.valid is True

    def test_frozen(self, gps: GPSFix) -> None:
        with pytest.raises(FrozenInstanceError):
            gps.valid = False  # type: ignore[misc]


class TestOpticalFlowReading:
    @pytest.fixture()
    def flow(self) -> OpticalFlowReading:
        return OpticalFlowReading(
            vx=0.5, vy=-0.3, quality=200, ground_distance=5.0, mono_ts=1.0,
        )

    def test_fields_accessible(self, flow: OpticalFlowReading) -> None:
        assert flow.vx == pytest.approx(0.5)
        assert flow.quality == 200

    def test_frozen(self, flow: OpticalFlowReading) -> None:
        with pytest.raises(FrozenInstanceError):
            flow.vx = 1.0  # type: ignore[misc]


class TestDetection:
    @pytest.fixture()
    def detection(self) -> Detection:
        return Detection(
            bbox=(0.1, 0.2, 0.3, 0.4),
            class_id=0,
            class_name="handgun",
            confidence=0.87,
        )

    def test_fields_accessible(self, detection: Detection) -> None:
        assert detection.bbox == (0.1, 0.2, 0.3, 0.4)
        assert detection.class_name == "handgun"
        assert detection.confidence == pytest.approx(0.87)

    def test_frozen(self, detection: Detection) -> None:
        with pytest.raises(FrozenInstanceError):
            detection.confidence = 0.5  # type: ignore[misc]

    def test_equality(self) -> None:
        d1 = Detection(bbox=(0.1, 0.2, 0.3, 0.4), class_id=0, class_name="handgun", confidence=0.87)
        d2 = Detection(bbox=(0.1, 0.2, 0.3, 0.4), class_id=0, class_name="handgun", confidence=0.87)
        assert d1 == d2


class TestNavState:
    @pytest.fixture()
    def nav(self) -> NavState:
        return NavState(
            position=(37.7749, -122.4194, 100.0),
            velocity=(0.0, 0.0, 0.0),
            attitude=(1.0, 0.0, 0.0, 0.0),
            position_uncertainty=(1.0, 1.0, 2.0),
            coordinate_frame="WGS84",
            mono_ts=1.0,
        )

    def test_fields_accessible(self, nav: NavState) -> None:
        assert nav.coordinate_frame == "WGS84"
        assert nav.position_uncertainty == (1.0, 1.0, 2.0)

    def test_frozen(self, nav: NavState) -> None:
        with pytest.raises(FrozenInstanceError):
            nav.coordinate_frame = "LOCAL_NED"  # type: ignore[misc]


class TestLinkStatus:
    def test_rssi_can_be_none(self) -> None:
        link = LinkStatus(
            connected=True, latency_ms=50.0, rssi_dbm=None,
            quality_pct=95.0, last_heartbeat_ts=1.0,
        )
        assert link.rssi_dbm is None

    def test_rssi_can_be_float(self) -> None:
        link = LinkStatus(
            connected=True, latency_ms=50.0, rssi_dbm=-65.0,
            quality_pct=95.0, last_heartbeat_ts=1.0,
        )
        assert link.rssi_dbm == pytest.approx(-65.0)


class TestActionDecision:
    def test_audit_trail(self) -> None:
        decision = ActionDecision(
            approved=True,
            action="TRACK_CLOSELY",
            authority="ROE_AUTONOMOUS",
            constraints={"max_duration_sec": 120},
            audit_id="audit-001",
        )
        assert decision.authority == "ROE_AUTONOMOUS"
        assert decision.audit_id == "audit-001"


# ---------------------------------------------------------------------------
# Protocol structural typing tests
# ---------------------------------------------------------------------------


class TestCameraHALProtocol:
    def test_conforming_class_passes_isinstance(self) -> None:
        class MyCamera:
            def open(self, config: dict[str, object]) -> None: ...
            def read_frame(self) -> Optional[np.ndarray]: return None
            def get_intrinsics(self) -> dict[str, object]: return {}
            def close(self) -> None: ...

        assert isinstance(MyCamera(), CameraHAL)

    def test_non_conforming_class_fails(self) -> None:
        class BadCamera:
            def open(self, config: dict[str, object]) -> None: ...

        assert not isinstance(BadCamera(), CameraHAL)


class TestAcceleratorHALProtocol:
    def test_conforming_class_passes(self) -> None:
        class MyAccelerator:
            def load_model(self, model_path: str, input_shapes: dict[str, object]) -> str:
                return "model-1"
            def infer(self, model_id: str, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
                return {}
            def get_capabilities(self) -> dict[str, object]:
                return {}
            def unload_model(self, model_id: str) -> None: ...

        assert isinstance(MyAccelerator(), AcceleratorHAL)


class TestOpticalFlowHALProtocol:
    def test_conforming_class_passes(self) -> None:
        class MyFlow:
            def open(self, config: dict[str, object]) -> None: ...
            def read(self) -> Optional[OpticalFlowReading]: return None
            def close(self) -> None: ...

        assert isinstance(MyFlow(), OpticalFlowHAL)


class TestAltimeterHALProtocol:
    def test_conforming_class_passes(self) -> None:
        class MyAlt:
            def open(self, config: dict[str, object]) -> None: ...
            def read_altitude(self) -> Optional[tuple[float, float]]: return None
            def close(self) -> None: ...

        assert isinstance(MyAlt(), AltimeterHAL)


class TestObjectDetectorProtocol:
    def test_conforming_class_passes(self) -> None:
        class MyDetector:
            def detect(self, frame: TimestampedFrame) -> list[Detection]: return []
            def get_class_names(self) -> list[str]: return []
            def swap_model(self, model_path: str) -> None: ...

        assert isinstance(MyDetector(), ObjectDetectorInterface)


class TestTrackerProtocol:
    def test_conforming_class_passes(self) -> None:
        class MyTracker:
            def update(self, detections: list[Detection], frame: TimestampedFrame) -> list[Track]:
                return []
            def reset(self) -> None: ...

        assert isinstance(MyTracker(), TrackerInterface)


class TestDecisionEngineProtocol:
    def test_conforming_class_passes(self) -> None:
        class MyEngine:
            def process(self, behaviors, tracks, nav_state, link_status):
                return None
            def get_state(self):
                return DecisionState.IDLE
            def set_mode(self, mode):
                pass
            def load_roe(self, roe_path):
                pass

        assert isinstance(MyEngine(), DecisionEngineInterface)
