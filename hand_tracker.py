"""
MediaPipe Hand Tracking wrapper.

Wraps mediapipe.solutions.hands to return structured landmark data
including pinch state, centroid position, and hand orientation angle.
"""

import math
from dataclasses import dataclass
from typing import List, Optional

import mediapipe as mp
import numpy as np

import config


@dataclass
class HandData:
    """Processed data for a single detected hand."""
    landmarks: list               # Raw MediaPipe landmarks
    pinch_distance: float         # Distance between thumb tip and index tip (normalized)
    centroid: np.ndarray          # Midpoint of thumb tip and index tip [x, y]
    orientation_angle: float      # Angle (radians) from centroid to wrist
    handedness: str               # "Left" or "Right"


class HandTracker:
    """Wraps MediaPipe Hands for structured landmark extraction."""

    # MediaPipe landmark indices
    THUMB_TIP = 4
    INDEX_TIP = 8
    WRIST = 0
    MIDDLE_MCP = 9

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=config.MAX_HANDS,
            min_detection_confidence=config.DETECTION_CONFIDENCE,
            min_tracking_confidence=config.TRACKING_CONFIDENCE,
            model_complexity=config.MODEL_COMPLEXITY,
        )
        self.mp_draw = mp.solutions.drawing_utils

    def process_frame(self, frame_rgb: np.ndarray) -> List[HandData]:
        """Process an RGB frame and return structured hand data.

        Args:
            frame_rgb: RGB image (H, W, 3) as numpy array.

        Returns:
            List of HandData for each detected hand.
        """
        results = self.hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            return []

        hands = []
        for hand_landmarks, handedness_info in zip(
            results.multi_hand_landmarks,
            results.multi_handedness,
        ):
            lm = hand_landmarks.landmark

            # Pinch distance (normalized coordinates)
            thumb = np.array([lm[self.THUMB_TIP].x, lm[self.THUMB_TIP].y])
            index = np.array([lm[self.INDEX_TIP].x, lm[self.INDEX_TIP].y])
            pinch_distance = np.linalg.norm(thumb - index)

            # Centroid (midpoint of thumb tip and index tip)
            centroid = (thumb + index) / 2.0

            # Orientation angle: vector from centroid to wrist
            wrist = np.array([lm[self.WRIST].x, lm[self.WRIST].y])
            vec = wrist - centroid
            orientation_angle = math.atan2(vec[1], vec[0])

            label = handedness_info.classification[0].label  # "Left" or "Right"

            hands.append(HandData(
                landmarks=lm,
                pinch_distance=pinch_distance,
                centroid=centroid,
                orientation_angle=orientation_angle,
                handedness=label,
            ))

        return hands

    def draw_landmarks(self, frame_bgr: np.ndarray, hands_data: List[HandData]):
        """Draw hand landmarks on a BGR frame for debugging/overlay."""
        # Re-run to get raw results for drawing (lightweight since cached internally)
        results = self.hands.process(
            frame_bgr[:, :, ::-1]  # BGR to RGB
        )
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame_bgr,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                )

    def close(self):
        """Release MediaPipe resources."""
        self.hands.close()
