"""
Win32 mouse / keyboard simulation for NX viewport control.

Uses ctypes to call Win32 APIs (SendInput) to simulate the mouse and
keyboard events that NX interprets as view manipulation:

    - Middle-mouse-button drag         → Orbit (rotate)
    - Shift + middle-mouse-button drag → Pan
    - Scroll wheel                     → Zoom

This avoids calling any NXOpen API and therefore works from any thread
or even from a completely separate process.
"""

import ctypes
import ctypes.wintypes as wt
import time

# ---------------------------------------------------------------------------
# Win32 constants
# ---------------------------------------------------------------------------

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_WHEEL = 0x0800

KEYEVENTF_KEYUP = 0x0002

VK_SHIFT = 0x10
VK_CONTROL = 0x11

WHEEL_DELTA = 120  # standard Windows scroll increment

# ---------------------------------------------------------------------------
# SendInput structures
# ---------------------------------------------------------------------------

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class _INPUTUnion(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("union", _INPUTUnion)]


_user32 = ctypes.windll.user32
_SendInput = _user32.SendInput
_SendInput.argtypes = [ctypes.c_uint, ctypes.POINTER(INPUT), ctypes.c_int]
_SendInput.restype = ctypes.c_uint


def _send(*inputs):
    """Send one or more INPUT events via SendInput."""
    n = len(inputs)
    arr = (INPUT * n)(*inputs)
    _SendInput(n, arr, ctypes.sizeof(INPUT))


def _mouse(dx=0, dy=0, flags=0, data=0):
    inp = INPUT()
    inp.type = INPUT_MOUSE
    inp.union.mi.dx = int(dx)
    inp.union.mi.dy = int(dy)
    inp.union.mi.dwFlags = flags
    inp.union.mi.mouseData = int(data)
    return inp


def _key(vk, up=False):
    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    inp.union.ki.wVk = vk
    inp.union.ki.dwFlags = KEYEVENTF_KEYUP if up else 0
    return inp


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def find_nx_window():
    """Return the HWND of the NX main window, or None."""
    result = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def _enum_cb(hwnd, _lparam):
        length = _user32.GetWindowTextLengthW(hwnd)
        if length > 0:
            buf = ctypes.create_unicode_buffer(length + 1)
            _user32.GetWindowTextW(hwnd, buf, length + 1)
            title = buf.value
            # NX window titles typically look like "NX 2406 - <part> - ..."
            if "NX" in title and _user32.IsWindowVisible(hwnd):
                result.append(hwnd)
        return True

    _user32.EnumWindows(_enum_cb, 0)
    return result[0] if result else None


def get_window_center(hwnd):
    """Return (x, y) of the client-area center in screen coordinates."""
    rect = wt.RECT()
    _user32.GetClientRect(hwnd, ctypes.byref(rect))
    pt = wt.POINT(rect.right // 2, rect.bottom // 2)
    _user32.ClientToScreen(hwnd, ctypes.byref(pt))
    return pt.x, pt.y


def set_cursor(x, y):
    _user32.SetCursorPos(int(x), int(y))


def set_foreground(hwnd):
    _user32.SetForegroundWindow(hwnd)


# ---------------------------------------------------------------------------
# High-level viewport actions
# ---------------------------------------------------------------------------

class NXViewportController:
    """Drives the NX viewport through simulated mouse input.

    Tracks gesture state so that middle-mouse-button presses and modifier
    keys are held across frames for smooth dragging.
    """

    # Internal states
    _IDLE = 0
    _ORBITING = 1
    _PANNING = 2

    def __init__(self):
        self._state = self._IDLE
        self._hwnd = None
        # Accumulators for sub-pixel movement
        self._accum_x = 0.0
        self._accum_y = 0.0

    def _ensure_window(self):
        if self._hwnd is None:
            self._hwnd = find_nx_window()
        return self._hwnd is not None

    def _begin_orbit(self):
        """Press MMB at the NX viewport center to start an orbit drag."""
        if not self._ensure_window():
            return
        cx, cy = get_window_center(self._hwnd)
        set_foreground(self._hwnd)
        set_cursor(cx, cy)
        time.sleep(0.01)
        _send(_mouse(flags=MOUSEEVENTF_MIDDLEDOWN))
        self._state = self._ORBITING

    def _begin_pan(self):
        """Press Shift + MMB at the NX viewport center to start a pan drag."""
        if not self._ensure_window():
            return
        cx, cy = get_window_center(self._hwnd)
        set_foreground(self._hwnd)
        set_cursor(cx, cy)
        time.sleep(0.01)
        _send(_key(VK_SHIFT), _mouse(flags=MOUSEEVENTF_MIDDLEDOWN))
        self._state = self._PANNING

    def _end_gesture(self):
        """Release all pressed buttons and modifiers."""
        if self._state == self._ORBITING:
            _send(_mouse(flags=MOUSEEVENTF_MIDDLEUP))
        elif self._state == self._PANNING:
            _send(_mouse(flags=MOUSEEVENTF_MIDDLEUP), _key(VK_SHIFT, up=True))
        self._state = self._IDLE
        self._accum_x = 0.0
        self._accum_y = 0.0

    def _move_mouse(self, dx, dy):
        """Send a relative mouse move, accumulating sub-pixel remainders."""
        self._accum_x += dx
        self._accum_y += dy
        ix = int(self._accum_x)
        iy = int(self._accum_y)
        if ix != 0 or iy != 0:
            _send(_mouse(dx=ix, dy=iy, flags=MOUSEEVENTF_MOVE))
            self._accum_x -= ix
            self._accum_y -= iy

    # ----- Public interface used by tracker_service -----

    def orbit(self, dx_pixels, dy_pixels):
        """Continue or begin an orbit gesture with the given pixel delta."""
        if self._state != self._ORBITING:
            self._end_gesture()
            self._begin_orbit()
        self._move_mouse(dx_pixels, dy_pixels)

    def pan(self, dx_pixels, dy_pixels):
        """Continue or begin a pan gesture with the given pixel delta."""
        if self._state != self._PANNING:
            self._end_gesture()
            self._begin_pan()
        self._move_mouse(dx_pixels, dy_pixels)

    def zoom(self, scroll_ticks):
        """Zoom by sending scroll wheel events.

        Ends any ongoing drag gesture first.
        """
        if self._state != self._IDLE:
            self._end_gesture()
        if abs(scroll_ticks) > 0.5:
            _send(_mouse(flags=MOUSEEVENTF_WHEEL,
                         data=int(scroll_ticks * WHEEL_DELTA)))

    def idle(self):
        """End any ongoing gesture (hand released)."""
        if self._state != self._IDLE:
            self._end_gesture()
