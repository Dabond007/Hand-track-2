"""
Tests for NXController with mocked NXOpen calls.

Verifies that the controller correctly dispatches pan, orbit, and zoom
commands, both in mock mode and with a mocked NXOpen module.
"""

import logging
import math
import sys
import os
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nx_controller import NXController


class TestNXControllerMockMode(unittest.TestCase):
    """Test NXController in mock mode (no NXOpen needed)."""

    def setUp(self):
        self.ctrl = NXController(mock_mode=True)

    def test_connected_in_mock_mode(self):
        self.assertTrue(self.ctrl.connected)

    def test_pan_logs(self):
        with self.assertLogs("nx_controller", level="INFO") as cm:
            self.ctrl.pan(10.0, -5.0)
        self.assertTrue(any("MOCK PAN" in msg for msg in cm.output))

    def test_orbit_logs(self):
        with self.assertLogs("nx_controller", level="INFO") as cm:
            self.ctrl.orbit(0.5)
        self.assertTrue(any("MOCK ORBIT" in msg for msg in cm.output))

    def test_zoom_logs(self):
        with self.assertLogs("nx_controller", level="INFO") as cm:
            self.ctrl.zoom(1.1)
        self.assertTrue(any("MOCK ZOOM" in msg for msg in cm.output))

    def test_zero_pan_skipped(self):
        # Should not log anything for zero values
        logger = logging.getLogger("nx_controller")
        with unittest.mock.patch.object(logger, "info") as mock_info:
            self.ctrl.pan(0.0, 0.0)
            # The mock_info should not have been called with PAN message
            for call in mock_info.call_args_list:
                self.assertNotIn("MOCK PAN", str(call))

    def test_zero_orbit_skipped(self):
        logger = logging.getLogger("nx_controller")
        with unittest.mock.patch.object(logger, "info") as mock_info:
            self.ctrl.orbit(0.0)
            for call in mock_info.call_args_list:
                self.assertNotIn("MOCK ORBIT", str(call))

    def test_zoom_factor_one_skipped(self):
        logger = logging.getLogger("nx_controller")
        with unittest.mock.patch.object(logger, "info") as mock_info:
            self.ctrl.zoom(1.0)
            for call in mock_info.call_args_list:
                self.assertNotIn("MOCK ZOOM", str(call))


class TestNXControllerWithMockedNXOpen(unittest.TestCase):
    """Test NXController with a mocked NXOpen module."""

    def _make_point3d(self, x=0.0, y=0.0, z=0.0):
        pt = MagicMock()
        pt.X = x
        pt.Y = y
        pt.Z = z
        return pt

    def _make_matrix3x3(self):
        """Create an identity rotation matrix mock."""
        m = MagicMock()
        m.Xx, m.Xy, m.Xz = 1.0, 0.0, 0.0
        m.Yx, m.Yy, m.Yz = 0.0, 1.0, 0.0
        m.Zx, m.Zy, m.Zz = 0.0, 0.0, 1.0
        return m

    def setUp(self):
        # Create a mock NXOpen module hierarchy
        self.mock_nxopen = MagicMock()
        self.mock_view = MagicMock()
        self.mock_view.Name = "TestView"
        self.mock_view.Origin = self._make_point3d(100.0, 200.0, 300.0)
        self.mock_view.Matrix = self._make_matrix3x3()
        self.mock_view.Scale = 1.0

        # Make Point3d constructor return a mock with the args as attributes
        def make_point3d(x, y, z):
            return self._make_point3d(x, y, z)
        self.mock_nxopen.Point3d = make_point3d

        # Make Matrix3x3 constructor return a writable mock
        def make_matrix3x3():
            return self._make_matrix3x3()
        self.mock_nxopen.Matrix3x3 = make_matrix3x3

        mock_part = MagicMock()
        mock_part.ModelingViews.WorkView = self.mock_view

        mock_session = MagicMock()
        mock_session.Parts.Work = mock_part

        self.mock_nxopen.Session.GetSession.return_value = mock_session

        # Inject mock NXOpen into sys.modules
        sys.modules["NXOpen"] = self.mock_nxopen

    def tearDown(self):
        if "NXOpen" in sys.modules:
            del sys.modules["NXOpen"]

    def test_connects_to_nx(self):
        ctrl = NXController(mock_mode=False)
        self.assertTrue(ctrl.connected)
        self.assertEqual(ctrl._view, self.mock_view)

    def test_pan_calls_set_origin(self):
        ctrl = NXController(mock_mode=False)
        ctrl.pan(10.0, -5.0)
        self.mock_view.SetOrigin.assert_called_once()
        pt = self.mock_view.SetOrigin.call_args[0][0]
        # dx=10 added to X=100, dy=-5 subtracted from Y=200 (inverted)
        self.assertAlmostEqual(pt.X, 110.0)
        self.assertAlmostEqual(pt.Y, 205.0)  # Y - (-5) = Y + 5
        self.assertAlmostEqual(pt.Z, 300.0)

    def test_orbit_calls_set_rotation_translation_scale(self):
        ctrl = NXController(mock_mode=False)
        ctrl.orbit(0.5)
        self.mock_view.SetRotationTranslationScale.assert_called_once()

    def test_zoom_calls_zoom_about_point(self):
        ctrl = NXController(mock_mode=False)
        ctrl.zoom(1.2)
        self.mock_view.ZoomAboutPoint.assert_called_once()
        args = self.mock_view.ZoomAboutPoint.call_args[0]
        self.assertAlmostEqual(args[0], 1.2)  # factor

    def test_reconnect(self):
        ctrl = NXController(mock_mode=False)
        ctrl._view = None
        self.assertFalse(ctrl.connected)

        ctrl.reconnect()
        self.assertTrue(ctrl.connected)


class TestNXControllerNoNXOpen(unittest.TestCase):
    """Test NXController when NXOpen is not available."""

    def test_not_connected_without_nxopen(self):
        # Ensure NXOpen is not in sys.modules
        saved = sys.modules.pop("NXOpen", None)
        try:
            ctrl = NXController(mock_mode=False)
            self.assertFalse(ctrl.connected)
        finally:
            if saved is not None:
                sys.modules["NXOpen"] = saved


if __name__ == "__main__":
    unittest.main()
