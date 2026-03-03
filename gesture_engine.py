"""
Gesture State Machine and Signal Processing.

Implements the gesture state machine (IDLE → SINGLE_HAND_ACTIVE → TWO_HAND_ZOOM)
and produces smoothed pan, orbit, and zoom signals from hand tracking data.
"""

import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

import numpy as np

import config
from hand_tracker import HandData


class GestureState(Enum):
    IDLE = auto()
    SINGLE_HAND_ACTIVE = auto()
    TWO_HAND_ZOOM = auto()


@dataclass
class GestureOutput:
    """Output signals from the gesture engine for one frame."""
    state: GestureState
    pan_dx: float = 0.0          # Pan delta X (screen-normalized, after sensitivity)
    pan_dy: float = 0.0          # Pan delta Y (screen-normalized, after sensitivity)
    orbit_delta: float = 0.0     # Orbit angle delta (radians, after sensitivity)
    zoom_factor: float = 1.0     # Zoom factor (>1 = zoom in, <1 = zoom out, 1 = no change)
    active_hand: Optional[str] = None  # Handedness of active single hand


@dataclass
class _HandState:
    """Internal tracking state for one hand."""
    pinched: bool = False
    centroid: Optional[np.ndarray] = None
    orientation_angle: float = 0.0
    lost_frames: int = 0


class GestureEngine:
    """Processes hand data into gesture commands using a state machine."""

    def __init__(self):
        self.state = GestureState.IDLE
        self._left = _HandState()
        self._right = _HandState()

        # Previous frame values for delta computation
        self._prev_centroid: Optional[np.ndarray] = None
        self._prev_angle: float = 0.0
        self._prev_zoom_dist: float = 0.0
        self._active_hand_label: Optional[str] = None

        # Smoothed signals (EMA)
        self._smooth_pan_x: float = 0.0
        self._smooth_pan_y: float = 0.0
        self._smooth_orbit: float = 0.0
        self._smooth_zoom: float = 0.0

        # Timing
        self._last_time: float = time.monotonic()

    def update(self, hands: List[HandData]) -> GestureOutput:
        """Process one frame of hand data and return gesture output.

        Args:
            hands: List of HandData from HandTracker.process_frame().

        Returns:
            GestureOutput with pan, orbit, and zoom deltas.
        """
        now = time.monotonic()
        dt = now - self._last_time
        self._last_time = now

        # Avoid division by zero or huge deltas on first frame
        if dt <= 0 or dt > 0.5:
            dt = 1.0 / 30.0

        # Organize hands by label
        left_data = None
        right_data = None
        for h in hands:
            if h.handedness == "Left":
                left_data = h
            elif h.handedness == "Right":
                right_data = h

        # Update pinch states with hysteresis
        self._update_pinch(self._left, left_data)
        self._update_pinch(self._right, right_data)

        # Determine state transitions
        left_pinched = self._left.pinched
        right_pinched = self._right.pinched

        if left_pinched and right_pinched:
            new_state = GestureState.TWO_HAND_ZOOM
        elif left_pinched or right_pinched:
            new_state = GestureState.SINGLE_HAND_ACTIVE
        else:
            new_state = GestureState.IDLE

        # Handle state transitions — reset references on change
        if new_state != self.state:
            self._on_state_change(new_state, left_pinched, right_pinched)

        self.state = new_state

        # Compute output based on state
        if self.state == GestureState.IDLE:
            return GestureOutput(state=self.state)

        if self.state == GestureState.SINGLE_HAND_ACTIVE:
            return self._process_single_hand(dt)

        if self.state == GestureState.TWO_HAND_ZOOM:
            return self._process_two_hand_zoom(dt)

        return GestureOutput(state=self.state)

    def _update_pinch(self, hand_state: _HandState, data: Optional[HandData]):
        """Update pinch detection for one hand with hysteresis."""
        if data is None:
            hand_state.lost_frames += 1
            if hand_state.lost_frames >= config.LOST_FRAMES_THRESHOLD:
                hand_state.pinched = False
                hand_state.centroid = None
            return

        hand_state.lost_frames = 0
        hand_state.centroid = data.centroid.copy()
        hand_state.orientation_angle = data.orientation_angle

        if not hand_state.pinched and data.pinch_distance < config.PINCH_THRESHOLD:
            hand_state.pinched = True
        elif hand_state.pinched and data.pinch_distance > (config.PINCH_THRESHOLD + config.PINCH_HYSTERESIS):
            hand_state.pinched = False

    def _on_state_change(self, new_state: GestureState, left_pinched: bool, right_pinched: bool):
        """Reset reference points when transitioning between states."""
        if new_state == GestureState.SINGLE_HAND_ACTIVE:
            # Determine which hand is active
            if left_pinched and not right_pinched:
                active = self._left
                self._active_hand_label = "Left"
            else:
                active = self._right
                self._active_hand_label = "Right"
            self._prev_centroid = active.centroid.copy() if active.centroid is not None else None
            self._prev_angle = active.orientation_angle
            self._smooth_pan_x = 0.0
            self._smooth_pan_y = 0.0
            self._smooth_orbit = 0.0

        elif new_state == GestureState.TWO_HAND_ZOOM:
            # Record initial distance between the two pinch centroids
            if self._left.centroid is not None and self._right.centroid is not None:
                self._prev_zoom_dist = np.linalg.norm(self._left.centroid - self._right.centroid)
            self._smooth_zoom = 0.0

        elif new_state == GestureState.IDLE:
            self._prev_centroid = None
            self._active_hand_label = None

    def _process_single_hand(self, dt: float) -> GestureOutput:
        """Compute pan and orbit signals for single-hand mode."""
        if self._active_hand_label == "Left":
            active = self._left
        else:
            active = self._right

        if active.centroid is None or self._prev_centroid is None:
            return GestureOutput(state=self.state, active_hand=self._active_hand_label)

        # Raw deltas
        raw_pan = active.centroid - self._prev_centroid
        raw_orbit = active.orientation_angle - self._prev_angle

        # Normalize orbit angle delta to [-pi, pi]
        raw_orbit = math.atan2(math.sin(raw_orbit), math.cos(raw_orbit))

        # Update previous values
        self._prev_centroid = active.centroid.copy()
        self._prev_angle = active.orientation_angle

        # Magnitudes
        pan_magnitude = np.linalg.norm(raw_pan)
        orbit_magnitude = abs(math.degrees(raw_orbit))

        # Apply dead zones
        if pan_magnitude < config.PAN_DEAD_ZONE:
            raw_pan = np.array([0.0, 0.0])
            pan_magnitude = 0.0
        if orbit_magnitude < config.ORBIT_DEAD_ZONE:
            raw_orbit = 0.0
            orbit_magnitude = 0.0

        # Orbit vs pan weighting
        epsilon = 1e-8
        total = orbit_magnitude + pan_magnitude + epsilon
        orbit_ratio = orbit_magnitude / total
        pan_weight = 1.0 - orbit_ratio
        orbit_weight = orbit_ratio

        # Apply sensitivity and dt scaling
        pan_x = raw_pan[0] * pan_weight * config.PAN_SENSITIVITY * dt
        pan_y = raw_pan[1] * pan_weight * config.PAN_SENSITIVITY * dt
        orbit = raw_orbit * orbit_weight * config.ORBIT_SENSITIVITY * dt

        # EMA smoothing
        alpha = config.EMA_ALPHA
        self._smooth_pan_x = alpha * pan_x + (1 - alpha) * self._smooth_pan_x
        self._smooth_pan_y = alpha * pan_y + (1 - alpha) * self._smooth_pan_y
        self._smooth_orbit = alpha * orbit + (1 - alpha) * self._smooth_orbit

        return GestureOutput(
            state=self.state,
            pan_dx=self._smooth_pan_x,
            pan_dy=self._smooth_pan_y,
            orbit_delta=self._smooth_orbit,
            active_hand=self._active_hand_label,
        )

    def _process_two_hand_zoom(self, dt: float) -> GestureOutput:
        """Compute zoom signal for two-hand mode."""
        if self._left.centroid is None or self._right.centroid is None:
            return GestureOutput(state=self.state, zoom_factor=1.0)

        current_dist = np.linalg.norm(self._left.centroid - self._right.centroid)
        raw_zoom_delta = current_dist - self._prev_zoom_dist
        self._prev_zoom_dist = current_dist

        # Dead zone
        if abs(raw_zoom_delta) < config.ZOOM_DEAD_ZONE:
            raw_zoom_delta = 0.0

        # Scale
        scaled = raw_zoom_delta * config.ZOOM_SENSITIVITY * dt

        # EMA smoothing
        alpha = config.EMA_ALPHA
        self._smooth_zoom = alpha * scaled + (1 - alpha) * self._smooth_zoom

        # Convert delta to a multiplicative factor centered on 1.0
        zoom_factor = 1.0 + self._smooth_zoom

        return GestureOutput(
            state=self.state,
            zoom_factor=zoom_factor,
        )

    def reset(self):
        """Reset all state to IDLE."""
        self.state = GestureState.IDLE
        self._left = _HandState()
        self._right = _HandState()
        self._prev_centroid = None
        self._prev_angle = 0.0
        self._prev_zoom_dist = 0.0
        self._active_hand_label = None
        self._smooth_pan_x = 0.0
        self._smooth_pan_y = 0.0
        self._smooth_orbit = 0.0
        self._smooth_zoom = 0.0
        self._last_time = time.monotonic()
