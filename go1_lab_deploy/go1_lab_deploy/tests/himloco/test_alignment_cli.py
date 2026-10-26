import argparse
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

PATH = Path(__file__).resolve().parents[2] / "scripts" / "check_alignment.py"
SPEC = importlib.util.spec_from_file_location("check_alignment_under_test", str(PATH))
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


class GuidedAlignmentTests(unittest.TestCase):
    def args(self, output, **changes):
        values = dict(guided="imu", duration=14., still_seconds=4.,
                      only=["roll_left_up", "pitch_nose_up"], output=Path(output),
                      config=Path("unused.json"), level_confirmed=False)
        values.update(changes)
        return argparse.Namespace(**values)

    def test_retake_only_selected_labels_with_longer_holds(self):
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch("builtins.input", return_value=""), \
                    mock.patch.object(CLI, "capture") as capture, \
                    mock.patch.object(CLI, "analyze") as analyze:
                CLI.guided(self.args(folder))
                self.assertEqual([c[0][0].capture for c in capture.call_args_list],
                                 ["roll_left_up", "pitch_nose_up"])
                self.assertTrue(all(c[0][0].still_seconds == 4. for c in capture.call_args_list))
                self.assertFalse(analyze.call_args[0][0].level_confirmed)

    def test_invalid_or_duplicate_selection_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            for labels in (["not_a_motion"], ["roll_left_up", "roll_left_up"]):
                with self.assertRaisesRegex(ValueError, "distinct labels"):
                    CLI.guided(self.args(folder, only=labels))

    def test_motion_interval_is_required(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "Still intervals"):
                CLI.guided(self.args(folder, duration=8.))

    def test_quit_does_not_create_receiver_or_analyze(self):
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch("builtins.input", return_value="q"), \
                    mock.patch.object(CLI, "capture") as capture, \
                    mock.patch.object(CLI, "analyze") as analyze:
                CLI.guided(self.args(folder))
                capture.assert_not_called()
                analyze.assert_not_called()


if __name__ == "__main__":
    unittest.main()
