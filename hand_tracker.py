"""
MediaPipe Hand Tracking wrapper.

Wraps the MediaPipe Tasks Vision HandLandmarker API to return structured
landmark data including pinch state, centroid position, and hand orientation angle.
"""

import math
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np

# Use attribute access for mediapipe Tasks API (lazy-loaded modules)
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

import config

# Hand connections for drawing (matches MediaPipe's HAND_CONNECTIONS)
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # index
    (0, 9), (9, 10), (10, 11), (11, 12),  # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
    (5, 9), (9, 13), (13, 17),             # palm
]

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")


@dataclass
class HandData:
    """Processed data for a single detected hand."""
    landmarks: list               # Raw MediaPipe landmarks (list of NormalizedLandmark)
    pinch_distance: float         # Distance between thumb tip and index tip (normalized)
    centroid: np.ndarray          # Midpoint of thumb tip and index tip [x, y]
    orientation_angle: float      # Angle (radians) from centroid to wrist
    handedness: str               # "Left" or "Right"


class HandTracker:
    """Wraps MediaPipe Tasks HandLandmarker for structured landmark extraction."""

    # MediaPipe landmark indices
    THUMB_TIP = 4
    INDEX_TIP = 8
    WRIST = 0
    MIDDLE_MCP = 9

    def __init__(self):
        if not os.path.isfile(MODEL_PATH):
            print(
                f"ERROR: Hand landmarker model not found at {MODEL_PATH}\n"
                "Download it with:\n"
                "  curl -L -o hand_landmarker.task "
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
                file=sys.stderr,
            )
            sys.exit(1)

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_hands=config.MAX_HANDS,
            min_hand_detection_confidence=config.DETECTION_CONFIDENCE,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=config.TRACKING_CONFIDENCE,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._frame_timestamp_ms = 0

    def process_frame(self, frame_rgb: np.ndarray) -> List[HandData]:
        """Process an RGB frame and return structured hand data.

        Args:
            frame_rgb: RGB image (H, W, 3) as numpy array.

        Returns:
            List of HandData for each detected hand.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Ensure monotonically increasing timestamps
        self._frame_timestamp_ms += 33  # ~30 fps interval
        results = self._landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)

        if not results.hand_landmarks:
            return []

        hands = []
        for hand_landmarks, handedness_list in zip(
            results.hand_landmarks,
            results.handedness,
        ):
            lm = hand_landmarks

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

            label = handedness_list[0].category_name  # "Left" or "Right"

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
        h, w = frame_bgr.shape[:2]

        for hand in hands_data:
            lm = hand.landmarks

            # Draw connections
            for start_idx, end_idx in _HAND_CONNECTIONS:
                x1, y1 = int(lm[start_idx].x * w), int(lm[start_idx].y * h)
                x2, y2 = int(lm[end_idx].x * w), int(lm[end_idx].y * h)
                cv2.line(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw landmark points
            for landmark in lm:
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame_bgr, (cx, cy), 4, (0, 0, 255), -1)

            # Highlight thumb tip and index tip for pinch visualization
            tx, ty = int(lm[self.THUMB_TIP].x * w), int(lm[self.THUMB_TIP].y * h)
            ix, iy = int(lm[self.INDEX_TIP].x * w), int(lm[self.INDEX_TIP].y * h)
            color = (0, 255, 255) if hand.pinch_distance < config.PINCH_THRESHOLD else (255, 0, 0)
            cv2.circle(frame_bgr, (tx, ty), 8, color, -1)
            cv2.circle(frame_bgr, (ix, iy), 8, color, -1)
            cv2.line(frame_bgr, (tx, ty), (ix, iy), color, 2)

    def close(self):
        """Release MediaPipe resources."""
        self._landmarker.close()
