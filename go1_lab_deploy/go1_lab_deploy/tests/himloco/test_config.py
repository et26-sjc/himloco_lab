import copy
import json
from pathlib import Path
import tempfile
import unittest

from go1_lab_deploy.himloco.config import load_config, readiness_errors, validate_config
from go1_lab_deploy.himloco.contract import TRAINING_JOINT_NAMES, POLICY_PROFILE
from fixtures import CONFIG_PATH, release_config


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = release_config()

    def test_real_lab_metadata_and_recorded_robot_evidence(self):
        validate_config(self.config, require_ready=True)
        self.assertEqual(self.config["schema_version"], 2)
        self.assertEqual(self.config["policy_profile"], POLICY_PROFILE)
        self.assertEqual(self.config["model"]["training_dof_names"], list(TRAINING_JOINT_NAMES))
        self.assertEqual(readiness_errors(self.config), [])

    def test_existing_readiness_gates_remain_required(self):
        for group, key, invalid in (("model", "sha256", None),
                                    ("model", "training_dof_names", None),
                                    ("alignment", "verified", False),
                                    ("alignment", "joint_mapping_reference", None),
                                    ("runtime", "safety_reviewed", False)):
            bad = copy.deepcopy(self.config)
            bad[group][key] = invalid
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config(bad, require_ready=True)

    def test_unknown_keys_and_contract_drift_rejected(self):
        cases = [(None, "foo", 1), ("model", "hip_reduction", 0.5),
                 ("runtime", "policy_dt", 0.002), ("runtime", "command_timeout_s", 0.2),
                 ("runtime", "state_timeout_s", 0.051), ("runtime", "rc_timeout_s", 0.2),
                 ("runtime", "max_cycle_lateness_s", 0.1), ("runtime", "history_warmup_frames", 5),
                 ("runtime", "startup_joint_margin_rad", .201),
                 ("runtime", "calibration_gain_ramp_s", 4),
                 ("runtime", "calibration_target_inset_rad", .02),
                 ("runtime", "safety_reviewed", "true"), ("runtime", "command_axis_signs", [1, 1, True]),
                 ("runtime", "command_deadzone", 1), ("runtime", "action_ramp_s", float("nan")),
                 ("alignment", "attitude_source", "xyzw"), (None, "source_commit", "main"),
                 (None, "schema_version", 1), (None, "policy_profile", "gym_legacy"),
                 ("model", "training_dof_names", list(reversed(TRAINING_JOINT_NAMES)))]
        for group, key, value in cases:
            bad = copy.deepcopy(self.config)
            (bad[group] if group else bad)[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config(bad)

    def test_relative_model_and_contract_paths_resolve_from_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            config_path = folder / "config.json"
            self.config["model"]["path"] = "models/policy.pt"
            contract = Path(self.config["deployment_contract"]["path"])
            (folder / "contract.json").write_bytes(contract.read_bytes())
            self.config["deployment_contract"]["path"] = "contract.json"
            config_path.write_text(json.dumps(self.config), encoding="utf-8")
            loaded = load_config(config_path)
            self.assertEqual(Path(loaded["model"]["path"]), folder / "models" / "policy.pt")
            self.assertEqual(Path(loaded["deployment_contract"]["path"]), folder / "contract.json")
            validate_config(loaded)

    def test_external_contract_hash_and_model_binding_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            for group, key, value in (("deployment_contract", "sha256", "0" * 64),
                                      ("model", "sha256", "0" * 64),
                                      (None, "source_commit", "0" * 40)):
                bad = copy.deepcopy(self.config)
                (bad[group] if group else bad)[key] = value
                path.write_text(json.dumps(bad))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    load_config(path)

    def test_duplicate_json_keys_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "duplicate.json"
            path.write_text('{"schema_version":1,"schema_version":2}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                load_config(path)

    def test_tracking_persistence_fields_and_backward_compatibility(self):
        for key, value in (("tracking_error_timeout_s", -0.01),
                           ("tracking_error_timeout_s", float("nan")),
                           ("tracking_error_timeout_s", True),
                           ("tracking_error_timeout_s", None),
                           ("max_instant_tracking_error_rad", .49),
                           ("max_instant_tracking_error_rad", float("inf")),
                           ("max_instant_tracking_error_rad", "1.0")):
            bad = copy.deepcopy(self.config)
            bad["runtime"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_config(bad)
        old = copy.deepcopy(self.config)
        old["runtime"].pop("tracking_error_timeout_s")
        old["runtime"].pop("max_instant_tracking_error_rad")
        validate_config(old, require_ready=True)


if __name__ == "__main__":
    unittest.main()
