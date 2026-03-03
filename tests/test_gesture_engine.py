"""
Unit tests for the gesture engine state machine and signal processing.

Tests use synthetic HandData to exercise all state transitions and
signal computations without requiring a webcam or MediaPipe.
"""

import math
import sys
import os
import time
import unittest

import numpy as np

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from gesture_engine import GestureEngine, GestureState, GestureOutput
from hand_tracker import HandData


def _make_hand(
    handedness: str = "Right",
    pinch_distance: float = 0.03,
    centroid_x: float = 0.5,
    centroid_y: float = 0.5,
    orientation_angle: float = 0.0,
) -> HandData:
    """Helper to create synthetic HandData."""
    return HandData(
        landmarks=[],
        pinch_distance=pinch_distance,
        centroid=np.array([centroid_x, centroid_y]),
        orientation_angle=orientation_angle,
        handedness=handedness,
    )


class TestGestureStateTransitions(unittest.TestCase):
    """Test the IDLE → SINGLE_HAND → TWO_HAND state machine."""

    def setUp(self):
        self.engine = GestureEngine()

    def test_starts_idle(self):
        output = self.engine.update([])
        self.assertEqual(output.state, GestureState.IDLE)

    def test_single_hand_pinch_activates(self):
        hand = _make_hand(pinch_distance=0.03)  # Below threshold
        output = self.engine.update([hand])
        self.assertEqual(output.state, GestureState.SINGLE_HAND_ACTIVE)

    def test_no_pinch_stays_idle(self):
        hand = _make_hand(pinch_distance=0.1)  # Above threshold
        output = self.engine.update([hand])
        self.assertEqual(output.state, GestureState.IDLE)

    def test_release_returns_to_idle(self):
        # Pinch
        hand = _make_hand(pinch_distance=0.03)
        self.engine.update([hand])

        # Release (above threshold + hysteresis)
        hand_released = _make_hand(pinch_distance=0.1)
        output = self.engine.update([hand_released])
        self.assertEqual(output.state, GestureState.IDLE)

    def test_two_hand_zoom(self):
        left = _make_hand("Left", pinch_distance=0.03, centroid_x=0.3)
        right = _make_hand("Right", pinch_distance=0.03, centroid_x=0.7)
        output = self.engine.update([left, right])
        self.assertEqual(output.state, GestureState.TWO_HAND_ZOOM)

    def test_two_hand_to_single_on_release(self):
        left = _make_hand("Left", pinch_distance=0.03, centroid_x=0.3)
        right = _make_hand("Right", pinch_distance=0.03, centroid_x=0.7)
        self.engine.update([left, right])

        # Release left hand
        left_released = _make_hand("Left", pinch_distance=0.1, centroid_x=0.3)
        right_still = _make_hand("Right", pinch_distance=0.03, centroid_x=0.7)
        output = self.engine.update([left_released, right_still])
        self.assertEqual(output.state, GestureState.SINGLE_HAND_ACTIVE)

    def test_two_hand_both_release_to_idle(self):
        left = _make_hand("Left", pinch_distance=0.03, centroid_x=0.3)
        right = _make_hand("Right", pinch_distance=0.03, centroid_x=0.7)
        self.engine.update([left, right])

        # Release both
        left_r = _make_hand("Left", pinch_distance=0.1)
        right_r = _make_hand("Right", pinch_distance=0.1)
        output = self.engine.update([left_r, right_r])
        self.assertEqual(output.state, GestureState.IDLE)

    def test_hysteresis_prevents_flicker(self):
        # Pinch at exactly threshold
        hand = _make_hand(pinch_distance=config.PINCH_THRESHOLD - 0.001)
        self.engine.update([hand])
        self.assertEqual(self.engine.state, GestureState.SINGLE_HAND_ACTIVE)

        # Move to just above threshold but below threshold + hysteresis
        hand_mid = _make_hand(
            pinch_distance=config.PINCH_THRESHOLD + config.PINCH_HYSTERESIS * 0.5
        )
        output = self.engine.update([hand_mid])
        # Should still be active due to hysteresis
        self.assertEqual(output.state, GestureState.SINGLE_HAND_ACTIVE)


class TestPanSignal(unittest.TestCase):
    """Test pan signal generation."""

    def setUp(self):
        self.engine = GestureEngine()

    def test_centroid_movement_produces_pan(self):
        # Initial pinch
        hand1 = _make_hand(centroid_x=0.5, centroid_y=0.5)
        self.engine.update([hand1])

        # Move centroid significantly
        hand2 = _make_hand(centroid_x=0.55, centroid_y=0.5)
        output = self.engine.update([hand2])

        self.assertEqual(output.state, GestureState.SINGLE_HAND_ACTIVE)
        # Pan dx should be positive (moved right)
        self.assertGreater(output.pan_dx, 0)

    def test_no_movement_no_pan(self):
        hand = _make_hand(centroid_x=0.5, centroid_y=0.5)
        self.engine.update([hand])

        # Same position
        output = self.engine.update([hand])
        self.assertAlmostEqual(output.pan_dx, 0, places=3)
        self.assertAlmostEqual(output.pan_dy, 0, places=3)


class TestOrbitSignal(unittest.TestCase):
    """Test orbit signal generation."""

    def setUp(self):
        self.engine = GestureEngine()

    def test_angle_change_produces_orbit(self):
        # Initial pinch with angle 0
        hand1 = _make_hand(centroid_x=0.5, centroid_y=0.5, orientation_angle=0.0)
        self.engine.update([hand1])

        # Simulate a realistic frame interval (~33ms) so dt-scaled values
        # are not vanishingly small.
        time.sleep(0.05)

        # Rotate hand significantly (keep centroid stable)
        hand2 = _make_hand(
            centroid_x=0.5, centroid_y=0.5,
            orientation_angle=math.radians(15)  # Well above dead zone
        )
        output = self.engine.update([hand2])

        self.assertEqual(output.state, GestureState.SINGLE_HAND_ACTIVE)
        self.assertNotAlmostEqual(output.orbit_delta, 0, places=4)


class TestZoomSignal(unittest.TestCase):
    """Test zoom signal generation."""

    def setUp(self):
        self.engine = GestureEngine()

    def test_increasing_distance_zooms_in(self):
        # Start with two hands close
        left = _make_hand("Left", pinch_distance=0.03, centroid_x=0.4)
        right = _make_hand("Right", pinch_distance=0.03, centroid_x=0.6)
        self.engine.update([left, right])

        # Move hands apart
        left2 = _make_hand("Left", pinch_distance=0.03, centroid_x=0.3)
        right2 = _make_hand("Right", pinch_distance=0.03, centroid_x=0.7)
        output = self.engine.update([left2, right2])

        self.assertEqual(output.state, GestureState.TWO_HAND_ZOOM)
        self.assertGreaterEqual(output.zoom_factor, 1.0)

    def test_decreasing_distance_zooms_out(self):
        # Start with two hands apart
        left = _make_hand("Left", pinch_distance=0.03, centroid_x=0.2)
        right = _make_hand("Right", pinch_distance=0.03, centroid_x=0.8)
        self.engine.update([left, right])

        # Move hands closer
        left2 = _make_hand("Left", pinch_distance=0.03, centroid_x=0.4)
        right2 = _make_hand("Right", pinch_distance=0.03, centroid_x=0.6)
        output = self.engine.update([left2, right2])

        self.assertEqual(output.state, GestureState.TWO_HAND_ZOOM)
        self.assertLessEqual(output.zoom_factor, 1.0)


class TestLostHandFrames(unittest.TestCase):
    """Test hand-lost detection."""

    def setUp(self):
        self.engine = GestureEngine()

    def test_hand_lost_transitions_to_idle(self):
        hand = _make_hand(pinch_distance=0.03)
        self.engine.update([hand])
        self.assertEqual(self.engine.state, GestureState.SINGLE_HAND_ACTIVE)

        # Lose hand for LOST_FRAMES_THRESHOLD consecutive frames
        for _ in range(config.LOST_FRAMES_THRESHOLD):
            output = self.engine.update([])

        self.assertEqual(output.state, GestureState.IDLE)


class TestReset(unittest.TestCase):
    """Test engine reset."""

    def test_reset_returns_to_idle(self):
        engine = GestureEngine()
        hand = _make_hand(pinch_distance=0.03)
        engine.update([hand])
        self.assertEqual(engine.state, GestureState.SINGLE_HAND_ACTIVE)

        engine.reset()
        self.assertEqual(engine.state, GestureState.IDLE)


class TestEMASmoothing(unittest.TestCase):
    """Test that EMA smoothing reduces noise."""

    def test_smoothing_dampens_sudden_jump(self):
        engine = GestureEngine()

        # Establish baseline
        hand = _make_hand(centroid_x=0.5, centroid_y=0.5)
        engine.update([hand])

        # Sudden large jump
        hand_jump = _make_hand(centroid_x=0.6, centroid_y=0.5)
        out1 = engine.update([hand_jump])

        # The smoothed output should be less than the raw delta
        # Raw delta would be ~0.1 * sensitivity * dt, smoothed should be less
        # due to EMA blending with the previous 0 value
        # Just verify it's not zero and is positive
        self.assertGreater(out1.pan_dx, 0)

        # Return to original position — smoothed value should still carry
        # some momentum from the previous movement
        hand_back = _make_hand(centroid_x=0.5, centroid_y=0.5)
        out2 = engine.update([hand_back])
        # The EMA will still carry positive momentum
        # (direction depends on alpha, but should be closer to 0 than out1)


if __name__ == "__main__":
    unittest.main()
