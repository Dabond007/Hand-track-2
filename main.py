"""
NX Hand Tracking 3D Mouse — Entry Point.

When run as an NX Journal, this script launches a lightweight subprocess
(tracker_service.py) that handles all heavy work (webcam, MediaPipe, OpenCV)
and drives the NX viewport via simulated Win32 mouse input.

No NXOpen API calls are made from Python — this avoids the main-thread
restriction that causes NX to freeze.

Usage:
    Standalone:    python main.py [--mock] [--overlay] [--no-mirror]
    NX Journal:    Execute from NX via File > Execute > NX Open
"""

import logging
import os
import subprocess
import sys
import threading
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Directory where this script lives
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_venv_python() -> str:
    """Locate the venv Python interpreter.

    Searches common locations relative to the project.
    """
    candidates = [
        # .venv next to the Hand-track-2 folder
        os.path.join(_SCRIPT_DIR, "..", ".venv", "Scripts", "python.exe"),
        # .venv inside the project folder
        os.path.join(_SCRIPT_DIR, ".venv", "Scripts", "python.exe"),
        # Linux/Mac variants
        os.path.join(_SCRIPT_DIR, "..", ".venv", "bin", "python"),
        os.path.join(_SCRIPT_DIR, ".venv", "bin", "python"),
    ]
    for path in candidates:
        resolved = os.path.normpath(path)
        if os.path.isfile(resolved):
            logger.info("Found venv Python: %s", resolved)
            return resolved

    # Fallback: hope "python" on PATH is the right one
    logger.warning("Could not find venv Python — falling back to 'python'")
    return "python"


def _detect_nx_journal() -> bool:
    """Return True when running inside Siemens NX as a journal."""
    try:
        import NXOpen  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# NX Journal mode — subprocess + pipe
# ---------------------------------------------------------------------------

def _run_nx_journal():
    """Launch the tracker subprocess.

    The subprocess handles everything — webcam, gesture detection, and
    viewport control via simulated Win32 mouse input.  No NXOpen calls
    are made from Python at all, so there is no main-thread requirement.
    """
    python_exe = _find_venv_python()
    tracker_script = os.path.join(_SCRIPT_DIR, "tracker_service.py")

    cmd = [python_exe, tracker_script, "--overlay"]
    logger.info("Launching tracker subprocess: %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        text=True,
    )

    def _log_output():
        """Forward subprocess stdout/stderr so the user can see status."""
        try:
            for line in proc.stderr:
                logger.info("tracker: %s", line.rstrip())
        except Exception:
            pass

    t = threading.Thread(target=_log_output, name="TrackerLog", daemon=True)
    t.start()
    logger.info(
        "Hand tracker running in background (PID %d). "
        "Press 'q' in the overlay window to stop.",
        proc.pid,
    )


# ---------------------------------------------------------------------------
# Standalone mode — everything in-process (original behaviour)
# ---------------------------------------------------------------------------

def _run_standalone(args):
    """Run the full tracking loop in-process (no NX required)."""
    import cv2
    import numpy as np

    import config
    from gesture_engine import GestureEngine, GestureState
    from hand_tracker import HandTracker, MODEL_PATH
    from nx_controller import NXController

    mirror = config.MIRROR_MODE and not args.no_mirror

    if not os.path.isfile(MODEL_PATH):
        logger.error(
            "Hand landmarker model not found at %s. "
            "Download it with:\n"
            "  curl -L -o hand_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
            MODEL_PATH,
        )
        sys.exit(1)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        logger.error("Failed to open webcam at index %d.", args.camera)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    tracker = HandTracker()
    engine = GestureEngine()
    nx = NXController(mock_mode=args.mock)

    if not nx.connected:
        logger.warning(
            "NX not connected. Gesture tracking will run but NX commands "
            "will not be sent. Use --mock for dry-run mode."
        )

    # State labels for HUD
    state_labels = {
        GestureState.IDLE: "IDLE",
        GestureState.SINGLE_HAND_ACTIVE: "SINGLE HAND",
        GestureState.TWO_HAND_ZOOM: "TWO HAND ZOOM",
    }

    nx_reconnect_interval = 10.0
    last_nx_reconnect = time.monotonic()
    frame_count = 0
    fps_start = time.monotonic()
    current_fps = 0.0

    logger.info("Starting main loop. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Webcam read failed. Exiting.")
                break

            if mirror:
                frame = cv2.flip(frame, 1)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hands = tracker.process_frame(frame_rgb)
            output = engine.update(hands)

            if output.state == GestureState.SINGLE_HAND_ACTIVE:
                nx.pan(output.pan_dx, output.pan_dy)
                nx.orbit(output.orbit_delta)
            elif output.state == GestureState.TWO_HAND_ZOOM:
                nx.zoom(output.zoom_factor)

            if not nx.connected:
                now = time.monotonic()
                if now - last_nx_reconnect > nx_reconnect_interval:
                    nx.reconnect()
                    last_nx_reconnect = now

            frame_count += 1
            elapsed = time.monotonic() - fps_start
            if elapsed >= 1.0:
                current_fps = frame_count / elapsed
                frame_count = 0
                fps_start = time.monotonic()

            if args.overlay:
                tracker.draw_landmarks(frame, hands)
                h, w = frame.shape[:2]
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                state_text = state_labels.get(output.state, "UNKNOWN")
                cv2.putText(frame, f"State: {state_text}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"FPS: {current_fps:.0f}", (w - 90, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.imshow("NX Hand Tracker", frame)

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


def main():
    if _detect_nx_journal():
        _run_nx_journal()
    else:
        import argparse
        import config

        parser = argparse.ArgumentParser(description="NX Hand Tracking 3D Mouse")
        parser.add_argument("--mock", action="store_true",
                            help="Run in mock mode (no NX connection)")
        parser.add_argument("--overlay", action="store_true",
                            help="Show webcam overlay with landmarks and HUD")
        parser.add_argument("--no-mirror", action="store_true",
                            help="Disable webcam mirroring")
        parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX,
                            help="Camera index (default: %(default)s)")
        args = parser.parse_args()
        _run_standalone(args)


if __name__ == "__main__":
    main()
