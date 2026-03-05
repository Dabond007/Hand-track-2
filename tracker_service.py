"""
Standalone hand-tracking process.

Runs webcam capture, MediaPipe hand tracking, and gesture detection in a
separate OS process.  Drives the NX viewport by simulating Win32 mouse
input (middle-button drag for orbit, Shift+MMB for pan, scroll for zoom).

Launched automatically by main.py when it detects an NX Journal environment.
Can also be run directly:  python tracker_service.py [--overlay]
"""

import logging
import os
import sys
import time

import cv2

import config
from gesture_engine import GestureEngine, GestureState
from hand_tracker import HandTracker, MODEL_PATH
from win32_mouse import NXViewportController

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Conversion factors:  gesture-engine units → mouse pixels / scroll ticks
PAN_PIXELS_SCALE = 1.0       # pan_dx/dy are already in ~pixel range (PAN_SENSITIVITY=500)
ORBIT_PIXELS_SCALE = 300.0   # radians → pixels of horizontal mouse drag
ZOOM_SCROLL_SCALE = 8.0      # (factor - 1.0) → scroll wheel ticks


def main():
    if not os.path.isfile(MODEL_PATH):
        logger.error("Model not found: %s", MODEL_PATH)
        return

    show_overlay = "--overlay" in sys.argv
    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    camera_index = int(positional[0]) if positional else config.CAMERA_INDEX
    mirror = config.MIRROR_MODE

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error("Cannot open camera %d", camera_index)
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    tracker = HandTracker()
    engine = GestureEngine()
    viewport = NXViewportController()

    logger.info("Tracker ready. Sending mouse input to NX viewport.")

    frame_count = 0
    fps_start = time.monotonic()
    current_fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Webcam read failed.")
                break

            if mirror:
                frame = cv2.flip(frame, 1)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hands = tracker.process_frame(frame_rgb)
            output = engine.update(hands)

            # Drive NX viewport via simulated mouse input
            if output.state == GestureState.SINGLE_HAND_ACTIVE:
                pan_px = abs(output.pan_dx) + abs(output.pan_dy)
                orbit_px = abs(output.orbit_delta) * ORBIT_PIXELS_SCALE

                if orbit_px > pan_px:
                    # Dominant motion is rotation → orbit (MMB drag)
                    viewport.orbit(
                        output.orbit_delta * ORBIT_PIXELS_SCALE,
                        0,
                    )
                else:
                    # Dominant motion is translation → pan (Shift+MMB drag)
                    viewport.pan(
                        output.pan_dx * PAN_PIXELS_SCALE,
                        output.pan_dy * PAN_PIXELS_SCALE,
                    )

            elif output.state == GestureState.TWO_HAND_ZOOM:
                zoom_delta = (output.zoom_factor - 1.0) * ZOOM_SCROLL_SCALE
                viewport.zoom(zoom_delta)

            else:
                viewport.idle()

            # FPS
            frame_count += 1
            elapsed = time.monotonic() - fps_start
            if elapsed >= 1.0:
                current_fps = frame_count / elapsed
                frame_count = 0
                fps_start = time.monotonic()

            # Overlay window
            if show_overlay:
                tracker.draw_landmarks(frame, hands)

                state_text = {
                    GestureState.IDLE: "IDLE",
                    GestureState.SINGLE_HAND_ACTIVE: "TRACKING",
                    GestureState.TWO_HAND_ZOOM: "ZOOM",
                }.get(output.state, "?")

                color = {
                    GestureState.IDLE: (128, 128, 128),
                    GestureState.SINGLE_HAND_ACTIVE: (0, 255, 0),
                    GestureState.TWO_HAND_ZOOM: (0, 200, 255),
                }.get(output.state, (255, 255, 255))

                cv2.putText(frame, f"{state_text}  FPS: {current_fps:.0f}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.imshow("Hand Tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                time.sleep(0.005)

    except KeyboardInterrupt:
        pass
    finally:
        viewport.idle()  # release any held buttons
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Tracker stopped.")


if __name__ == "__main__":
    main()
