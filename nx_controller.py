"""
NXOpen View Manipulation Controller.

Provides pan, orbit, and zoom commands to the active Siemens NX 2406 view.
Includes a MOCK_MODE for development/testing without NX running.

IMPORTANT: The NXOpen API calls below are approximations based on the NXOpen
Python API documentation. For production use:
  1. Record a journal in NX 2406 of manual pan, orbit, and zoom operations.
  2. Extract the actual Python API calls from the journal.
  3. Update the methods below accordingly.
"""

import logging
import math

logger = logging.getLogger(__name__)


class NXController:
    """Controls the NX view via NXOpen Python API."""

    def __init__(self, mock_mode: bool = False):
        """Initialize NX controller.

        Args:
            mock_mode: If True, log commands instead of calling NXOpen.
        """
        self.mock_mode = mock_mode
        self._session = None
        self._view = None

        if not mock_mode:
            self._connect()

    def _connect(self):
        """Connect to the active NX session."""
        try:
            import NXOpen
            self._session = NXOpen.Session.GetSession()
            work_part = self._session.Parts.Work
            if work_part is None:
                logger.error("No work part open in NX session.")
                return
            # Get the current layout's view
            layout = work_part.Layouts.Current
            self._view = layout.GetView()
            logger.info("Connected to NX session. View: %s", self._view.Name)
        except ImportError:
            logger.error(
                "NXOpen module not available. "
                "Run from NX's Python environment or use mock_mode=True."
            )
            self._session = None
            self._view = None
        except Exception:
            logger.exception("Failed to connect to NX session.")
            self._session = None
            self._view = None

    @property
    def connected(self) -> bool:
        """Return True if connected to NX (or in mock mode)."""
        return self.mock_mode or self._view is not None

    def pan(self, dx: float, dy: float):
        """Pan (translate) the NX view.

        Args:
            dx: Horizontal pan delta (screen pixels).
            dy: Vertical pan delta (screen pixels).
        """
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return

        if self.mock_mode:
            logger.info("MOCK PAN: dx=%.3f, dy=%.3f", dx, dy)
            return

        if self._view is None:
            return

        try:
            # NXOpen View.Pan(deltaX, deltaY) — screen-relative
            # NOTE: Verify exact API signature from NX 2406 journal recording
            self._view.Pan(dx, dy)
        except Exception:
            logger.exception("NX Pan failed.")

    def orbit(self, angle_delta: float):
        """Orbit (rotate) the NX view around the screen center.

        Args:
            angle_delta: Rotation angle in radians.
        """
        if abs(angle_delta) < 1e-8:
            return

        if self.mock_mode:
            logger.info("MOCK ORBIT: angle=%.4f rad (%.2f deg)",
                        angle_delta, math.degrees(angle_delta))
            return

        if self._view is None:
            return

        try:
            import NXOpen
            # Rotate about the Z-axis (screen normal) through the view center.
            # NOTE: Verify exact API from journal recording. The actual call
            # may use View.Rotate(origin, axis, angle) or
            # Display.Camera manipulation.
            origin = NXOpen.Point3d(0.0, 0.0, 0.0)
            axis = NXOpen.Vector3d(0.0, 0.0, 1.0)
            self._view.Rotate(origin, axis, angle_delta)
        except Exception:
            logger.exception("NX Orbit failed.")

    def zoom(self, factor: float):
        """Zoom the NX view.

        Args:
            factor: Zoom factor. >1 = zoom in, <1 = zoom out, 1 = no change.
        """
        if abs(factor - 1.0) < 1e-6:
            return

        if self.mock_mode:
            logger.info("MOCK ZOOM: factor=%.4f", factor)
            return

        if self._view is None:
            return

        try:
            # NXOpen View.Zoom(factor)
            # NOTE: Verify exact API from journal recording.
            self._view.Zoom(factor)
        except Exception:
            logger.exception("NX Zoom failed.")

    def refresh(self):
        """Force a view refresh/redraw."""
        if self.mock_mode:
            return

        if self._view is None:
            return

        try:
            # Attempt to trigger a redraw
            self._view.Regenerate()
        except Exception:
            # Some NX versions may not have Regenerate; silently ignore
            pass

    def reconnect(self):
        """Attempt to reconnect to NX session."""
        if not self.mock_mode:
            logger.info("Attempting NX reconnection...")
            self._connect()
