"""
NX Hand Tracking 3D Mouse — Entry Point.

Main loop: capture frame → process hands → update gesture state → send NX commands.

Usage:
    Standalone:    python main.py [--mock] [--overlay] [--no-mirror]
    NX Journal:    Execute from NX via File > Execute > NX Open
"""

import argparse
import logging
import os
import sys
import threading
import time

import cv2
import numpy as np

import config
from gesture_engine import GestureEngine, GestureState
from hand_tracker import HandTracker, MODEL_PATH
from nx_controller import NXController

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Flag used to signal the tracking thread to stop
_stop_event = threading.Event()

# State labels for HUD
_STATE_LABELS = {
    GestureState.IDLE: "IDLE",
    GestureState.SINGLE_HAND_ACTIVE: "SINGLE HAND",
    GestureState.TWO_HAND_ZOOM: "TWO HAND ZOOM",
}


def draw_hud(frame: np.ndarray, output, fps: float):
    """Draw a status overlay on the frame."""
    h, w = frame.shape[:2]
    # Semi-transparent bar at the top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    state_text = _STATE_LABELS.get(output.state, "UNKNOWN")
    color = {
        GestureState.IDLE: (128, 128, 128),
        GestureState.SINGLE_HAND_ACTIVE: (0, 255, 0),
        GestureState.TWO_HAND_ZOOM: (0, 200, 255),
    }.get(output.state, (255, 255, 255))

    cv2.putText(frame, f"State: {state_text}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    details = []
    if output.state == GestureState.SINGLE_HAND_ACTIVE:
        details.append(f"Pan: ({output.pan_dx:+.2f}, {output.pan_dy:+.2f})")
        details.append(f"Orbit: {output.orbit_delta:+.4f} rad")
        if output.active_hand:
            details.append(f"Hand: {output.active_hand}")
    elif output.state == GestureState.TWO_HAND_ZOOM:
        details.append(f"Zoom: {output.zoom_factor:.4f}")

    detail_str = "  |  ".join(details)
    cv2.putText(frame, detail_str, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 90, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def _detect_nx_journal() -> bool:
    """Return True when running inside Siemens NX as a journal."""
    try:
        import NXOpen  # noqa: F401
        return True
    except ImportError:
        return False


def _tracking_loop(args, mirror: bool):
    """Core capture → process → command loop.

    Runs until ``_stop_event`` is set or the webcam fails.
    """
    # Verify model file exists before initializing components
    if not os.path.isfile(MODEL_PATH):
        logger.error(
            "Hand landmarker model not found at %s. "
            "Download it with:\n"
            "  curl -L -o hand_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
            MODEL_PATH,
        )
        return

    # Initialize components
    logger.info("Initializing webcam (index %d)...", args.camera)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        logger.error("Failed to open webcam at index %d.", args.camera)
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    logger.info("Initializing hand tracker...")
    tracker = HandTracker()

    logger.info("Initializing gesture engine...")
    engine = GestureEngine()

    logger.info("Initializing NX controller (mock=%s)...", args.mock)
    nx = NXController(mock_mode=args.mock)

    if not nx.connected:
        logger.warning(
            "NX not connected. Gesture tracking will run but NX commands "
            "will not be sent. Use --mock for dry-run mode."
        )

    # NX reconnection timer
    nx_reconnect_interval = 10.0  # seconds
    last_nx_reconnect = time.monotonic()

    # FPS tracking
    frame_count = 0
    fps_start = time.monotonic()
    current_fps = 0.0

    logger.info("Starting tracking loop. Press 'q' in overlay to quit.")

    try:
        while not _stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                logger.error("Webcam read failed. Exiting.")
                break

            # Mirror if configured
            if mirror:
                frame = cv2.flip(frame, 1)

            # Convert BGR to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process hand tracking
            hands = tracker.process_frame(frame_rgb)

            # Update gesture engine
            output = engine.update(hands)

            # Send commands to NX
            if output.state == GestureState.SINGLE_HAND_ACTIVE:
                nx.pan(output.pan_dx, output.pan_dy)
                nx.orbit(output.orbit_delta)
            elif output.state == GestureState.TWO_HAND_ZOOM:
                nx.zoom(output.zoom_factor)

            # Periodically try to reconnect NX if not connected
            if not nx.connected:
                now = time.monotonic()
                if now - last_nx_reconnect > nx_reconnect_interval:
                    nx.reconnect()
                    last_nx_reconnect = now

            # FPS calculation
            frame_count += 1
            elapsed = time.monotonic() - fps_start
            if elapsed >= 1.0:
                current_fps = frame_count / elapsed
                frame_count = 0
                fps_start = time.monotonic()

            # Overlay window
            if args.overlay:
                tracker.draw_landmarks(frame, hands)
                draw_hud(frame, output, current_fps)
                cv2.imshow("NX Hand Tracker", frame)

            # Check for quit (waitKey also pumps the OpenCV event loop)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("Quit requested.")
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        logger.info("Shutting down...")
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Done.")


def _parse_args():
    """Parse CLI arguments (returns defaults when run as NX journal)."""
    parser = argparse.ArgumentParser(description="NX Hand Tracking 3D Mouse")
    parser.add_argument("--mock", action="store_true",
                        help="Run in mock mode (no NX connection)")
    parser.add_argument("--overlay", action="store_true",
                        help="Show webcam overlay with landmarks and HUD")
    parser.add_argument("--no-mirror", action="store_true",
                        help="Disable webcam mirroring")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX,
                        help="Camera index (default: %(default)s)")
    # When NX runs a journal it may pass extra args; ignore them.
    args, _unknown = parser.parse_known_args()
    return args


def main():
    args = _parse_args()
    mirror = config.MIRROR_MODE and not args.no_mirror

    if _detect_nx_journal():
        # Running inside NX — launch the tracking loop on a daemon thread
        # so the NX UI thread is not blocked.
        logger.info("NX Journal detected — starting hand tracker in background thread.")
        _stop_event.clear()
        t = threading.Thread(target=_tracking_loop, args=(args, mirror), daemon=True)
        t.start()
        # Return immediately so NX remains responsive.
        # The daemon thread will be cleaned up when NX exits.
    else:
        # Standalone mode — run on the main thread as before.
        _tracking_loop(args, mirror)


if __name__ == "__main__":
    main()
