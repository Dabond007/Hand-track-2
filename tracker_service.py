"""
Standalone hand-tracking process.

Runs webcam capture, MediaPipe hand tracking, and gesture detection in a
separate OS process.  Writes JSON command lines to stdout so the NX journal
(main.py) can read them without loading any heavy libraries itself.

Not intended to be run by the user directly — launched automatically by
main.py when it detects an NX Journal environment.
"""

import json
import sys
import time

import cv2
import numpy as np

import config
from gesture_engine import GestureEngine, GestureState
from hand_tracker import HandTracker, MODEL_PATH


def _send(obj: dict):
    """Write a JSON line to stdout and flush immediately."""
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main():
    import os

    if not os.path.isfile(MODEL_PATH):
        _send({"type": "error", "msg": f"Model not found: {MODEL_PATH}"})
        return

    camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else config.CAMERA_INDEX
    mirror = config.MIRROR_MODE
    show_overlay = "--overlay" in sys.argv

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        _send({"type": "error", "msg": f"Cannot open camera {camera_index}"})
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    tracker = HandTracker()
    engine = GestureEngine()

    _send({"type": "ready"})

    frame_count = 0
    fps_start = time.monotonic()
    current_fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                _send({"type": "error", "msg": "Webcam read failed"})
                break

            if mirror:
                frame = cv2.flip(frame, 1)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hands = tracker.process_frame(frame_rgb)
            output = engine.update(hands)

            # Send command to NX journal
            if output.state == GestureState.SINGLE_HAND_ACTIVE:
                _send({
                    "type": "cmd",
                    "action": "single",
                    "pan_dx": output.pan_dx,
                    "pan_dy": output.pan_dy,
                    "orbit_delta": output.orbit_delta,
                })
            elif output.state == GestureState.TWO_HAND_ZOOM:
                _send({
                    "type": "cmd",
                    "action": "zoom",
                    "zoom_factor": output.zoom_factor,
                })

            # FPS
            frame_count += 1
            elapsed = time.monotonic() - fps_start
            if elapsed >= 1.0:
                current_fps = frame_count / elapsed
                frame_count = 0
                fps_start = time.monotonic()

            # Optional overlay window (useful for debugging)
            if show_overlay:
                tracker.draw_landmarks(frame, hands)
                cv2.putText(frame, f"FPS: {current_fps:.0f}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Hand Tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # Small sleep to cap frame rate and reduce CPU usage
                time.sleep(0.005)

    except (KeyboardInterrupt, BrokenPipeError):
        pass
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        _send({"type": "stopped"})


if __name__ == "__main__":
    main()
