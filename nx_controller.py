"""
NXOpen View Manipulation Controller.

Provides pan, orbit, and zoom commands to the active Siemens NX 2406 view.
Includes a MOCK_MODE for development/testing without NX running.

API calls derived from NX 2406 journal recordings:
  - Orbit: view.SetRotationTranslationScale(matrix3x3, translation, scale)
  - Zoom:  view.ZoomAboutPoint(factor, scaleAboutPoint, viewCenter)
  - Pan:   view.SetOrigin(point3d)
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
        self._NXOpen = None  # Cache the NXOpen module

        if not mock_mode:
            self._connect()

    def _connect(self):
        """Connect to the active NX session."""
        try:
            import NXOpen
            self._NXOpen = NXOpen
            self._session = NXOpen.Session.GetSession()
            work_part = self._session.Parts.Work
            if work_part is None:
                logger.error("No work part open in NX session.")
                return
            self._view = work_part.ModelingViews.WorkView
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
        """Pan (translate) the NX view by shifting the view origin.

        Uses view.SetOrigin() as recorded in NX journals.

        Args:
            dx: Horizontal pan delta (screen-normalized, after sensitivity).
            dy: Vertical pan delta (screen-normalized, after sensitivity).
        """
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return

        if self.mock_mode:
            logger.info("MOCK PAN: dx=%.3f, dy=%.3f", dx, dy)
            return

        if self._view is None:
            return

        try:
            NXOpen = self._NXOpen
            # Get current origin and shift by the pan delta.
            # dx/dy are in screen-normalized units scaled by PAN_SENSITIVITY,
            # so they map directly to view-space displacement.
            origin = self._view.Origin
            new_origin = NXOpen.Point3d(
                origin.X + dx,
                origin.Y - dy,  # Screen Y is inverted relative to model Y
                origin.Z,
            )
            self._view.SetOrigin(new_origin)
        except Exception:
            logger.exception("NX Pan failed.")

    def orbit(self, angle_delta: float):
        """Orbit (rotate) the NX view around the screen Z-axis.

        Uses view.SetRotationTranslationScale() as recorded in NX journals.
        Applies an incremental rotation about the view's Z-axis (screen normal).

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
            NXOpen = self._NXOpen
            # Get current view transform
            rot = self._view.Matrix
            translation = self._view.Origin
            scale = self._view.Scale

            # Build incremental rotation about the view Z-axis
            c = math.cos(angle_delta)
            s = math.sin(angle_delta)

            # Multiply current rotation matrix by Z-rotation:
            #   Rz = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
            #   new_R = Rz * current_R
            new_rot = NXOpen.Matrix3x3()
            new_rot.Xx = c * rot.Xx - s * rot.Yx
            new_rot.Xy = c * rot.Xy - s * rot.Yy
            new_rot.Xz = c * rot.Xz - s * rot.Yz
            new_rot.Yx = s * rot.Xx + c * rot.Yx
            new_rot.Yy = s * rot.Xy + c * rot.Yy
            new_rot.Yz = s * rot.Xz + c * rot.Yz
            new_rot.Zx = rot.Zx
            new_rot.Zy = rot.Zy
            new_rot.Zz = rot.Zz

            # Preserve the current translation as a Point3d
            trans_pt = NXOpen.Point3d(translation.X, translation.Y, translation.Z)
            self._view.SetRotationTranslationScale(new_rot, trans_pt, scale)
        except Exception:
            logger.exception("NX Orbit failed.")

    def zoom(self, factor: float):
        """Zoom the NX view about its center.

        Uses view.ZoomAboutPoint() as recorded in NX journals.

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
            NXOpen = self._NXOpen
            # Zoom about the screen center (0, 0, 0)
            center = NXOpen.Point3d(0.0, 0.0, 0.0)
            self._view.ZoomAboutPoint(factor, center, center)
        except Exception:
            logger.exception("NX Zoom failed.")

    def reconnect(self):
        """Attempt to reconnect to NX session."""
        if not self.mock_mode:
            logger.info("Attempting NX reconnection...")
            self._connect()
