"""
Configuration and tunable parameters for NX Hand Tracking 3D Mouse.

All constants are centralized here for easy tuning.
"""

# --- Pinch Detection ---
PINCH_THRESHOLD = 0.05          # Normalized distance for pinch detection
PINCH_HYSTERESIS = 0.01         # Prevent flicker at threshold boundary

# --- Smoothing ---
EMA_ALPHA = 0.3                 # Exponential moving average factor (0-1, higher = less smoothing)

# --- Dead Zones ---
PAN_DEAD_ZONE = 0.005           # Minimum centroid movement to register as pan
ORBIT_DEAD_ZONE = 1.0           # Minimum angle change (degrees) to register as orbit
ZOOM_DEAD_ZONE = 0.01           # Minimum pinch distance change to register as zoom

# --- Sensitivity ---
PAN_SENSITIVITY = 500.0         # Screen pixels per normalized unit
ORBIT_SENSITIVITY = 2.0         # Radians multiplier
ZOOM_SENSITIVITY = 0.02         # Scale factor per normalized distance unit

# --- Classification ---
ORBIT_VS_PAN_THRESHOLD = 0.6   # Ratio threshold for orbit vs pan weighting

# --- MediaPipe ---
MAX_HANDS = 2
DETECTION_CONFIDENCE = 0.7
TRACKING_CONFIDENCE = 0.6

# --- Camera ---
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MIRROR_MODE = True              # Mirror webcam feed so left hand movement = left pan

# --- Tracking ---
LOST_FRAMES_THRESHOLD = 5      # Consecutive lost frames before transitioning to IDLE
