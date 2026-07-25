"""Tests for core data types — enums, frozen dataclasses, and protocols.

Verifies:
1. Enum members exist and have correct values
2. Frozen dataclasses are truly immutable (including container fields)
3. Protocol structural typing works (isinstance checks)
4. Types with numpy arrays use identity equality (eq=False)
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from aerialmind.core.protocols import (
    GPSHAL,
    IMUHAL,
    AcceleratorHAL,
    AltimeterHAL,
    BehaviorAnalyzerInterface,
    CameraHAL,
    DecisionEngineInterface,
    NavigationEKFInterface,
    ObjectDetectorInterface,
    OpticalFlowHAL,
    PoseEstimatorInterface,
    TelemetryInterface,
    TrackerInterface,
    VIOEngineInterface,
)
from aerialmind.core.types import (
    ActionConstraints,
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
    ThreatSignal,
    TimestampedFrame,
    TimestampedIMU,
    Track,
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

    def test_identity_equality(self, frame: TimestampedFrame) -> None:
        assert frame == frame
        other = TimestampedFrame(
            frame=np.zeros((720, 1280, 3), dtype=np.uint8),
            mono_ts=1.0, seq_id=1, width=1280, height=720,
        )
        assert frame != other

    def test_hashable(self, frame: TimestampedFrame) -> None:
        assert isinstance(hash(frame), int)
        assert {frame}


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


class TestPoseResult:
    @pytest.fixture()
    def pose(self) -> PoseResult:
        return PoseResult(
            keypoints=np.zeros((17, 3), dtype=np.float32),
            detection=Detection(
                bbox=(0.1, 0.2, 0.3, 0.4), class_id=0,
                class_name="person", confidence=0.9,
            ),
        )

    def test_fields_accessible(self, pose: PoseResult) -> None:
        assert pose.keypoints.shape == (17, 3)
        assert pose.detection.class_name == "person"

    def test_frozen(self, pose: PoseResult) -> None:
        with pytest.raises(FrozenInstanceError):
            pose.detection = Detection(  # type: ignore[misc]
                bbox=(0.0, 0.0, 0.0, 0.0), class_id=1,
                class_name="other", confidence=0.1,
            )

    def test_identity_equality(self, pose: PoseResult) -> None:
        assert pose == pose
        other = PoseResult(
            keypoints=np.zeros((17, 3), dtype=np.float32),
            detection=Detection(
                bbox=(0.1, 0.2, 0.3, 0.4), class_id=0,
                class_name="person", confidence=0.9,
            ),
        )
        assert pose != other

    def test_hashable(self, pose: PoseResult) -> None:
        assert isinstance(hash(pose), int)


class TestTrack:
    @pytest.fixture()
    def track(self) -> Track:
        return Track(
            track_id=1, bbox=(0.1, 0.2, 0.3, 0.4),
            velocity=(1.0, -0.5), age=30,
            class_id=0, class_name="person", confidence=0.85,
        )

    def test_fields_accessible(self, track: Track) -> None:
        assert track.track_id == 1
        assert track.age == 30
        assert track.velocity == (1.0, -0.5)

    def test_frozen(self, track: Track) -> None:
        with pytest.raises(FrozenInstanceError):
            track.age = 31  # type: ignore[misc]

    def test_equality(self) -> None:
        t1 = Track(track_id=1, bbox=(0.1, 0.2, 0.3, 0.4), velocity=(1.0, -0.5),
                    age=30, class_id=0, class_name="person", confidence=0.85)
        t2 = Track(track_id=1, bbox=(0.1, 0.2, 0.3, 0.4), velocity=(1.0, -0.5),
                    age=30, class_id=0, class_name="person", confidence=0.85)
        assert t1 == t2


class TestBehaviorEvent:
    @pytest.fixture()
    def event(self) -> BehaviorEvent:
        return BehaviorEvent(
            event_type="fight",
            confidence=0.75,
            involved_tracks=(1, 3, 7),
            location_frame=(0.5, 0.6),
            mono_ts=1.0,
        )

    def test_fields_accessible(self, event: BehaviorEvent) -> None:
        assert event.event_type == "fight"
        assert event.involved_tracks == (1, 3, 7)

    def test_frozen(self, event: BehaviorEvent) -> None:
        with pytest.raises(FrozenInstanceError):
            event.confidence = 0.9  # type: ignore[misc]

    def test_involved_tracks_is_truly_immutable(self, event: BehaviorEvent) -> None:
        with pytest.raises(AttributeError):
            event.involved_tracks.append(99)  # type: ignore[attr-defined]

    def test_hashable(self, event: BehaviorEvent) -> None:
        assert isinstance(hash(event), int)


class TestNavState:
    @pytest.fixture()
    def nav(self) -> NavState:
        return NavState(
            position=(37.7749, -122.4194, 100.0),
            velocity=(0.0, 0.0, 0.0),
            attitude_wxyz=(1.0, 0.0, 0.0, 0.0),
            position_uncertainty=(1.0, 1.0, 2.0),
            coordinate_frame="WGS84",
            mono_ts=1.0,
        )

    def test_fields_accessible(self, nav: NavState) -> None:
        assert nav.coordinate_frame == "WGS84"
        assert nav.position_uncertainty == (1.0, 1.0, 2.0)
        assert nav.attitude_wxyz == (1.0, 0.0, 0.0, 0.0)

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


class TestThreatSignal:
    @pytest.fixture()
    def signal(self) -> ThreatSignal:
        return ThreatSignal(track_id=7, type="weapon", confidence=0.87)

    def test_fields_accessible(self, signal: ThreatSignal) -> None:
        assert signal.track_id == 7
        assert signal.type == "weapon"
        assert signal.confidence == pytest.approx(0.87)

    def test_frozen(self, signal: ThreatSignal) -> None:
        with pytest.raises(FrozenInstanceError):
            signal.confidence = 0.5  # type: ignore[misc]

    def test_hashable(self, signal: ThreatSignal) -> None:
        assert isinstance(hash(signal), int)


class TestActionConstraints:
    def test_all_fields_default_none(self) -> None:
        c = ActionConstraints()
        assert c.max_duration_sec is None
        assert c.max_deviation_m is None
        assert c.min_altitude_m is None

    def test_partial_fields(self) -> None:
        c = ActionConstraints(max_duration_sec=120.0, min_altitude_m=30.0)
        assert c.max_duration_sec == pytest.approx(120.0)
        assert c.max_deviation_m is None
        assert c.min_altitude_m == pytest.approx(30.0)

    def test_frozen(self) -> None:
        c = ActionConstraints(max_duration_sec=120.0)
        with pytest.raises(FrozenInstanceError):
            c.max_duration_sec = 60.0  # type: ignore[misc]

    def test_hashable(self) -> None:
        c = ActionConstraints(max_duration_sec=120.0)
        assert isinstance(hash(c), int)


class TestThreatAssessment:
    @pytest.fixture()
    def assessment(self) -> ThreatAssessment:
        return ThreatAssessment(
            level=ThreatLevel.HIGH,
            score=75.0,
            threats=(
                ThreatSignal(track_id=7, type="weapon", confidence=0.87),
            ),
            mono_ts=1.0,
        )

    def test_fields_accessible(self, assessment: ThreatAssessment) -> None:
        assert assessment.level == ThreatLevel.HIGH
        assert assessment.score == pytest.approx(75.0)
        assert len(assessment.threats) == 1
        assert assessment.threats[0].track_id == 7

    def test_frozen(self, assessment: ThreatAssessment) -> None:
        with pytest.raises(FrozenInstanceError):
            assessment.score = 50.0  # type: ignore[misc]

    def test_threats_tuple_is_immutable(self, assessment: ThreatAssessment) -> None:
        with pytest.raises(AttributeError):
            assessment.threats.append(  # type: ignore[attr-defined]
                ThreatSignal(track_id=2, type="behavior", confidence=0.5),
            )

    def test_hashable(self, assessment: ThreatAssessment) -> None:
        assert isinstance(hash(assessment), int)


class TestRecommendation:
    @pytest.fixture()
    def rec(self) -> Recommendation:
        return Recommendation(
            action="TRACK_CLOSELY",
            priority=1,
            rationale="Weapon detected in NE sector",
            constraints=ActionConstraints(
                max_duration_sec=120.0, max_deviation_m=200.0,
            ),
            roe_rule_id="C1",
        )

    def test_fields_accessible(self, rec: Recommendation) -> None:
        assert rec.action == "TRACK_CLOSELY"
        assert rec.constraints.max_duration_sec == pytest.approx(120.0)
        assert rec.roe_rule_id == "C1"

    def test_frozen(self, rec: Recommendation) -> None:
        with pytest.raises(FrozenInstanceError):
            rec.priority = 2  # type: ignore[misc]

    def test_constraints_frozen(self, rec: Recommendation) -> None:
        with pytest.raises(FrozenInstanceError):
            rec.constraints.max_duration_sec = 60.0  # type: ignore[misc]

    def test_hashable(self, rec: Recommendation) -> None:
        assert isinstance(hash(rec), int)


class TestActionDecision:
    @pytest.fixture()
    def decision(self) -> ActionDecision:
        return ActionDecision(
            approved=True,
            action="TRACK_CLOSELY",
            authority="ROE_AUTONOMOUS",
            constraints=ActionConstraints(max_duration_sec=120.0),
            audit_id="audit-001",
        )

    def test_audit_trail(self, decision: ActionDecision) -> None:
        assert decision.authority == "ROE_AUTONOMOUS"
        assert decision.audit_id == "audit-001"

    def test_frozen(self, decision: ActionDecision) -> None:
        with pytest.raises(FrozenInstanceError):
            decision.approved = False  # type: ignore[misc]

    def test_constraints_frozen(self, decision: ActionDecision) -> None:
        with pytest.raises(FrozenInstanceError):
            decision.constraints.max_duration_sec = 99.0  # type: ignore[misc]

    def test_hashable(self, decision: ActionDecision) -> None:
        assert isinstance(hash(decision), int)


# ---------------------------------------------------------------------------
# Protocol structural typing tests
# ---------------------------------------------------------------------------


class TestCameraHALProtocol:
    def test_conforming_class_passes_isinstance(self) -> None:
        class MyCamera:
            def open(self, config: dict[str, object]) -> None: ...
            def read_frame(self) -> np.ndarray | None: return None
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


class TestIMUHALProtocol:
    def test_conforming_class_passes(self) -> None:
        class MyIMU:
            def open(self, config: dict[str, object]) -> None: ...
            def read(self) -> TimestampedIMU | None: return None
            def get_noise_params(self) -> dict[str, object]: return {}
            def close(self) -> None: ...

        assert isinstance(MyIMU(), IMUHAL)

    def test_non_conforming_class_fails(self) -> None:
        class BadIMU:
            def open(self, config: dict[str, object]) -> None: ...

        assert not isinstance(BadIMU(), IMUHAL)


class TestGPSHALProtocol:
    def test_conforming_class_passes(self) -> None:
        class MyGPS:
            def open(self, config: dict[str, object]) -> None: ...
            def read(self) -> GPSFix | None: return None
            def close(self) -> None: ...

        assert isinstance(MyGPS(), GPSHAL)

    def test_non_conforming_class_fails(self) -> None:
        class BadGPS:
            def read(self) -> GPSFix | None: return None

        assert not isinstance(BadGPS(), GPSHAL)


class TestOpticalFlowHALProtocol:
    def test_conforming_class_passes(self) -> None:
        class MyFlow:
            def open(self, config: dict[str, object]) -> None: ...
            def read(self) -> OpticalFlowReading | None: return None
            def close(self) -> None: ...

        assert isinstance(MyFlow(), OpticalFlowHAL)


class TestAltimeterHALProtocol:
    def test_conforming_class_passes(self) -> None:
        class MyAlt:
            def open(self, config: dict[str, object]) -> None: ...
            def read_altitude(self) -> tuple[float, float] | None: return None
            def close(self) -> None: ...

        assert isinstance(MyAlt(), AltimeterHAL)


class TestObjectDetectorProtocol:
    def test_conforming_class_passes(self) -> None:
        class MyDetector:
            def detect(self, frame: TimestampedFrame) -> list[Detection]: return []
            def get_class_names(self) -> list[str]: return []
            def swap_model(self, model_path: str) -> None: ...

        assert isinstance(MyDetector(), ObjectDetectorInterface)


class TestPoseEstimatorProtocol:
    def test_conforming_class_passes(self) -> None:
        class MyPoseEstimator:
            def estimate(self, frame: TimestampedFrame,
                         detections: list[Detection]) -> list[PoseResult]:
                return []

        assert isinstance(MyPoseEstimator(), PoseEstimatorInterface)

    def test_non_conforming_class_fails(self) -> None:
        class BadPose:
            pass

        assert not isinstance(BadPose(), PoseEstimatorInterface)


class TestTrackerProtocol:
    def test_conforming_class_passes(self) -> None:
        class MyTracker:
            def update(self, detections: list[Detection], frame: TimestampedFrame) -> list[Track]:
                return []
            def reset(self) -> None: ...

        assert isinstance(MyTracker(), TrackerInterface)


class TestBehaviorAnalyzerProtocol:
    def test_conforming_class_passes(self) -> None:
        class MyAnalyzer:
            def analyze(self, tracks: list[Track],
                        poses: list[PoseResult]) -> list[BehaviorEvent]:
                return []

        assert isinstance(MyAnalyzer(), BehaviorAnalyzerInterface)

    def test_non_conforming_class_fails(self) -> None:
        class BadAnalyzer:
            pass

        assert not isinstance(BadAnalyzer(), BehaviorAnalyzerInterface)


class TestVIOEngineProtocol:
    def test_conforming_class_passes(self) -> None:
        class MyVIO:
            def initialize(self, camera_intrinsics: dict[str, object],
                           imu_noise: dict[str, object]) -> None: ...
            def process_frame(self, frame: TimestampedFrame,
                              imu_readings: list[TimestampedIMU]) -> NavState | None:
                return None
            def get_tracking_quality(self) -> float: return 1.0
            def reset(self) -> None: ...

        assert isinstance(MyVIO(), VIOEngineInterface)

    def test_non_conforming_class_fails(self) -> None:
        class BadVIO:
            def initialize(self, camera_intrinsics: dict[str, object],
                           imu_noise: dict[str, object]) -> None: ...

        assert not isinstance(BadVIO(), VIOEngineInterface)


class TestNavigationEKFProtocol:
    def test_conforming_class_passes(self) -> None:
        class MyEKF:
            def predict(self, imu: TimestampedIMU) -> NavState: ...
            def update_gps(self, gps: GPSFix) -> NavState: ...
            def update_vio(self, vio_pose: NavState) -> NavState: ...
            def update_baro(self, altitude: float, mono_ts: float) -> NavState: ...
            def update_optical_flow(self, flow: OpticalFlowReading) -> NavState: ...
            def update_altimeter(self, altitude: float, mono_ts: float) -> NavState: ...
            def get_state(self) -> NavState: ...

        assert isinstance(MyEKF(), NavigationEKFInterface)

    def test_non_conforming_class_fails(self) -> None:
        class BadEKF:
            def predict(self, imu: TimestampedIMU) -> NavState: ...

        assert not isinstance(BadEKF(), NavigationEKFInterface)


class TestDecisionEngineProtocol:
    def test_conforming_class_passes(self) -> None:
        class MyEngine:
            def process(self, behaviors: list[BehaviorEvent], tracks: list[Track],
                        nav_state: NavState, link_status: LinkStatus) -> ActionDecision: ...
            def get_state(self) -> DecisionState: return DecisionState.IDLE
            def set_mode(self, mode: OperatingMode) -> None: ...
            def load_roe(self, roe_path: str) -> None: ...

        assert isinstance(MyEngine(), DecisionEngineInterface)


class TestTelemetryProtocol:
    def test_conforming_class_passes(self) -> None:
        class MyTelemetry:
            def send_scene_report(self, detections: list[Detection],
                                  tracks: list[Track],
                                  behaviors: list[BehaviorEvent],
                                  nav_state: NavState) -> bool:
                return True
            def send_alert(self, assessment: ThreatAssessment,
                           recommendation: Recommendation) -> bool:
                return True
            def receive_command(self) -> dict[str, object] | None:
                return None

        assert isinstance(MyTelemetry(), TelemetryInterface)

    def test_non_conforming_class_fails(self) -> None:
        class BadTelemetry:
            def send_scene_report(self) -> bool: return True

        assert not isinstance(BadTelemetry(), TelemetryInterface)
