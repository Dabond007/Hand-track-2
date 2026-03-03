# NX Hand Tracking 3D Mouse

A Python application that uses a webcam and MediaPipe Hand Tracking to provide gesture-based 3D view manipulation (orbit, pan, zoom) in Siemens NX 2406 via the NXOpen Python API.

## Gesture Controls

| Gesture | Action |
|---------|--------|
| **Single-hand pinch** (thumb + index) — move hand | Pan the NX view |
| **Single-hand pinch** — rotate hand around pinch point | Orbit/rotate the NX view |
| **Two-hand pinch** — move hands apart/together | Zoom in/out |
| **Release pinch** | Deactivate tracking (free hand repositioning) |

Pan and orbit are weighted and can blend simultaneously based on the dominant signal.

## Requirements

- Python 3.8+
- Webcam
- Siemens NX 2406 (for NXOpen integration; not required for testing)

## Installation

```bash
pip install -r requirements.txt
```

NXOpen is provided by the NX installation and does not need to be pip-installed.

## Usage

### Standalone (with NX running)

Run from NX's Python environment or with NX's Python on PATH:

```bash
python main.py
```

### Mock Mode (without NX)

Test gesture tracking without NX installed. Commands are logged to console:

```bash
python main.py --mock --overlay
```

### As an NX Journal

1. Open NX 2406
2. Go to **File > Execute > NX Open**
3. Select `main.py`

### Command-Line Options

| Flag | Description |
|------|-------------|
| `--mock` | Run without NX connection (log commands to console) |
| `--overlay` | Show webcam window with hand landmarks and status HUD |
| `--no-mirror` | Disable webcam mirroring (default: mirrored) |
| `--camera N` | Webcam index (default: 0) |

Press **q** to quit when the overlay window is active.

## Configuration

All tunable parameters are in `config.py`:

- **Pinch detection**: threshold, hysteresis
- **Smoothing**: EMA alpha
- **Dead zones**: pan, orbit, zoom minimum deltas
- **Sensitivity**: pan, orbit, zoom multipliers
- **MediaPipe**: detection/tracking confidence, model complexity
- **Camera**: index, resolution, mirror mode

## NXOpen API Note

The NXOpen API calls in `nx_controller.py` are approximations. For production use:

1. In NX 2406, start recording a journal (**Developer > Journal > Record**, Python language)
2. Manually orbit, pan, and zoom the view
3. Stop recording and inspect the generated Python code
4. Update `nx_controller.py` with the exact API calls from the journal

## Running Tests

```bash
python -m pytest tests/ -v
```

Or with unittest:

```bash
python -m unittest discover tests -v
```

Tests cover the gesture state machine, signal processing, and NX controller (mocked) — no webcam or NX installation required.

## Project Structure

```
nx-hand-track/
├── main.py                 # Entry point, main loop
├── hand_tracker.py         # MediaPipe wrapper
├── gesture_engine.py       # State machine, signal processing
├── nx_controller.py        # NXOpen API integration
├── config.py               # All tunable parameters
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── tests/
    ├── test_gesture_engine.py   # Unit tests for state machine & signals
    └── test_mock_nx.py          # Tests with mocked NXOpen calls
```
