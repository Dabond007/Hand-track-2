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

    def setUp(self):
        # Create a mock NXOpen module hierarchy
        self.mock_nxopen = MagicMock()
        self.mock_view = MagicMock()
        self.mock_view.Name = "TestView"

        mock_layout = MagicMock()
        mock_layout.GetView.return_value = self.mock_view

        mock_part = MagicMock()
        mock_part.Layouts.Current = mock_layout

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

    def test_pan_calls_view(self):
        ctrl = NXController(mock_mode=False)
        ctrl.pan(10.0, -5.0)
        self.mock_view.Pan.assert_called_once_with(10.0, -5.0)

    def test_orbit_calls_view(self):
        ctrl = NXController(mock_mode=False)
        ctrl.orbit(0.5)
        self.mock_view.Rotate.assert_called_once()
        args = self.mock_view.Rotate.call_args[0]
        self.assertAlmostEqual(args[2], 0.5)  # angle

    def test_zoom_calls_view(self):
        ctrl = NXController(mock_mode=False)
        ctrl.zoom(1.2)
        self.mock_view.Zoom.assert_called_once_with(1.2)

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
