"""Offline CLI/package tests: real release metadata, mocked communication."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

from fixtures import CONFIG_PATH, release_config

DEPLOY_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = DEPLOY_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, str(DEPLOY_ROOT / "scripts" / (name + ".py")))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


cli = load_script("deploy_policy")
bundle = load_script("build_deployment_bundle")
bridge = load_script("verify_bridge_build")
EXAMPLE = CONFIG_PATH


class FakePolicy:
    calls = []

    def __init__(self, path, expected_sha256):
        self.calls.append(("load", path, expected_sha256))

    def __call__(self, history):
        if history.shape != (1, 270) or history.dtype != np.float32:
            raise ValueError("invalid mock input")
        self.calls.append(("infer",))
        return np.zeros(12, dtype=np.float32)


class CliTests(unittest.TestCase):
    def setUp(self):
        FakePolicy.calls = []
        self.runner = types.ModuleType("go1_lab_deploy.himloco.runner")
        self.runner.run_deployment = mock.Mock()
        self.runner.observe_states = mock.Mock()

    def invoke(self, args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = cli.main(args)
        return result, stdout.getvalue(), stderr.getvalue()

    def test_help_does_not_import_torch_or_lcm(self):
        source = ("import runpy,sys\n"
                  "target=sys.argv[1]\nsys.argv=[target,'--help']\n"
                  "try:\n    runpy.run_path(target,run_name='__main__')\n"
                  "except SystemExit as exc:\n    assert exc.code == 0\n"
                  "assert 'torch' not in sys.modules\nassert 'lcm' not in sys.modules\n")
        completed = subprocess.run([sys.executable, "-c", source,
                                    str(DEPLOY_ROOT / "scripts" / "deploy_policy.py")],
                                   capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_real_configuration_inspection_never_loads_model_or_lcm(self):
        with mock.patch("go1_lab_deploy.himloco.policy.JitPolicy") as policy:
            result, output, error = self.invoke(["--config", str(EXAMPLE), "--inspect-config"])
        self.assertEqual(result, 0, error)
        data = json.loads(output)
        self.assertFalse(data["model_loaded"])
        self.assertFalse(data["motor_commands_published"])
        self.assertEqual(data["readiness_blockers"], [])
        policy.assert_not_called()

    def test_default_mode_is_offline_preflight(self):
        with mock.patch("go1_lab_deploy.himloco.policy.JitPolicy", FakePolicy), \
                mock.patch.dict(sys.modules, {self.runner.__name__: self.runner}):
            result, output, _ = self.invoke(["--config", str(EXAMPLE)])
        self.assertEqual(result, 0)
        self.assertFalse(json.loads(output)["motor_commands_published"])
        self.assertEqual([entry[0] for entry in FakePolicy.calls], ["load", "infer"])
        self.runner.run_deployment.assert_not_called()
        self.runner.observe_states.assert_not_called()

    def test_run_requires_verified_configuration_before_loading_policy(self):
        config = release_config()
        config["alignment"]["verified"] = False
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unverified.json"
            path.write_text(json.dumps(config))
            with mock.patch("go1_lab_deploy.himloco.policy.JitPolicy", FakePolicy), \
                    mock.patch.dict(sys.modules, {self.runner.__name__: self.runner}):
                result, _, error = self.invoke(["--config", str(path), "--run"])
        self.assertEqual(result, 2)
        self.assertIn("not ready", error.lower())
        self.assertEqual(FakePolicy.calls, [])
        self.runner.run_deployment.assert_not_called()

    def test_observe_never_loads_policy(self):
        with mock.patch("go1_lab_deploy.himloco.policy.JitPolicy", FakePolicy), \
                mock.patch.dict(sys.modules, {self.runner.__name__: self.runner}):
            result, _, _ = self.invoke(["--config", str(EXAMPLE), "--observe", "--duration", "0.1"])
        self.assertEqual(result, 0)
        self.assertEqual(FakePolicy.calls, [])
        self.runner.observe_states.assert_called_once()
        self.assertEqual(self.runner.observe_states.call_args.kwargs["duration"], 0.1)
        self.runner.run_deployment.assert_not_called()

    def test_verified_run_loads_and_checks_policy_before_runner(self):
        # Hardware evidence is inherited verbatim from the release. The mocked
        # runner below never creates LCM and sends no commands.
        config = release_config()
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "fixture.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            def run_checked(*args, **kwargs):
                self.assertEqual([entry[0] for entry in FakePolicy.calls], ["load", "infer"])
                self.assertEqual(kwargs["log_path"], "fixture.jsonl")
            self.runner.run_deployment.side_effect = run_checked
            with mock.patch("go1_lab_deploy.himloco.policy.JitPolicy", FakePolicy), \
                    mock.patch.dict(sys.modules, {self.runner.__name__: self.runner}):
                result, _, error = self.invoke(["--config", str(config_path), "--run", "--log", "fixture.jsonl"])
        self.assertEqual(result, 0, error)
        self.runner.run_deployment.assert_called_once()

    def test_invalid_policy_output_does_not_enter_runner(self):
        with mock.patch("go1_lab_deploy.himloco.policy.JitPolicy", return_value=lambda _: np.zeros(13)), \
                mock.patch.dict(sys.modules, {self.runner.__name__: self.runner}):
            result, _, error = self.invoke(["--config", str(EXAMPLE)])
        self.assertEqual(result, 2)
        self.assertIn("twelve finite", error)
        self.runner.run_deployment.assert_not_called()

    def test_model_contract_binding_failure_never_enters_runner(self):
        config = release_config()
        config["model"]["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad-hash.json"
            path.write_text(json.dumps(config))
            result, _, error = self.invoke(["--config", str(path), "--check"])
        self.assertEqual(result, 2)
        self.assertIn("sha", error.lower())
        self.runner.run_deployment.assert_not_called()

    def test_mutually_exclusive_modes_rejected(self):
        with self.assertRaises(SystemExit) as error, contextlib.redirect_stderr(io.StringIO()):
            cli.main(["--config", str(EXAMPLE), "--check", "--run"])
        self.assertEqual(error.exception.code, 2)

    def test_invalid_duration_rejected(self):
        with self.assertRaises(SystemExit) as error, contextlib.redirect_stderr(io.StringIO()):
            cli.main(["--config", str(EXAMPLE), "--observe", "--duration", "nan"])
        self.assertEqual(error.exception.code, 2)


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "entry.py").write_text("# fixture\n", encoding="utf-8")
        (self.source / "lcm_position").write_bytes(b"old executable must not travel")
        (self.source / "accidental.pt").write_bytes(b"old model must not travel")
        self.config = release_config()
        self.model = self.root / "source-policy.pt"
        self.model.write_bytes(Path(self.config["model"]["path"]).read_bytes())
        self.config_path = self.root / "source-config.json"
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")

    def build(self):
        return bundle.build_bundle(self.config_path, self.model, self.root / "release", self.source)

    def test_bundle_model_path_portable_and_hashes_verified(self):
        release = self.build()
        config = json.loads((release / "himloco.json").read_text(encoding="utf-8"))
        self.assertEqual(config["model"]["path"], "models/policy.pt")
        self.assertEqual(config["model"]["sha256"], self.config["model"]["sha256"])
        self.assertGreaterEqual(bundle.verify_bundle(release), 4)
        descriptor = config["deployment_contract"]
        self.assertFalse(Path(descriptor["path"]).is_absolute())
        self.assertTrue((release / descriptor["path"]).is_file())
        self.assertFalse((release / "go1_lab_deploy" / "lcm_position").exists())
        self.assertFalse((release / "go1_lab_deploy" / "accidental.pt").exists())
        self.assertEqual(json.loads(self.config_path.read_text())["model"]["path"],
                         self.config["model"]["path"])

    def test_hash_mismatch_rejects_before_output_creation(self):
        self.model.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.build()
        self.assertFalse((self.root / "release").exists())

    def test_utf8_bom_configuration_can_be_packaged(self):
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8-sig")
        self.assertGreaterEqual(bundle.verify_bundle(self.build()), 4)

    def test_modified_release_file_fails_verification(self):
        release = self.build()
        (release / "go1_lab_deploy" / "entry.py").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "integrity failure"):
            bundle.verify_bundle(release)

    def test_existing_release_never_overwritten(self):
        self.build()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.build()

    def test_manifest_traversal_rejected(self):
        release = self.build()
        path = release / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["files"]["../source-policy.pt"] = self.config["model"]["sha256"]
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "outside"):
            bundle.verify_bundle(release)


class BridgeBuildTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in bridge.SOURCE_FILES:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture source " + name, encoding="utf-8")
        self.binary = self.root / "lcm_position_him"
        self.digest = bridge.source_digest(self.root)
        self.binary.write_bytes(b"synthetic binary HIM_BRIDGE protocol=" + self.digest.encode("ascii"))
        self.receipt = self.root / "lcm_position_him.sha256"
        self.write_receipt()
        self.run = mock.Mock(return_value=types.SimpleNamespace(
            stdout="HIM_BRIDGE protocol=20261023 source_sha256=" + self.digest + "\n"))

    def write_receipt(self):
        self.receipt.write_text(hashlib.sha256(self.binary.read_bytes()).hexdigest() +
                                "  lcm_position_him\n", encoding="ascii")

    def test_current_source_and_receipt_allow_version_query(self):
        self.assertIn(self.digest, bridge.verify_bridge(self.binary, self.root, self.run))
        self.run.assert_called_once()
        self.assertEqual(self.run.call_args.args[0][1], "--version")
        self.assertEqual(self.run.call_args.kwargs["stdin"], subprocess.DEVNULL)
        kwargs = self.run.call_args.kwargs
        self.assertNotIn("capture_output", kwargs)
        self.assertNotIn("text", kwargs)
        self.assertEqual(kwargs["stdout"], subprocess.PIPE)
        self.assertEqual(kwargs["stderr"], subprocess.PIPE)
        self.assertTrue(kwargs["universal_newlines"])

    def test_stale_source_rejected_before_executing_binary(self):
        (self.root / bridge.SOURCE_FILES[0]).write_text("updated", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "current HIM"):
            bridge.verify_bridge(self.binary, self.root, self.run)
        self.run.assert_not_called()

    def test_corrupt_binary_rejected_before_execution(self):
        self.binary.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "checksum"):
            bridge.verify_bridge(self.binary, self.root, self.run)
        self.run.assert_not_called()

    def test_renamed_old_bridge_rejected_before_execution(self):
        self.binary.write_bytes(b"old WTW binary with no HIM version support")
        self.write_receipt()
        with self.assertRaisesRegex(ValueError, "current HIM"):
            bridge.verify_bridge(self.binary, self.root, self.run)
        self.run.assert_not_called()

    def test_wrong_protocol_version_rejected(self):
        self.run.return_value.stdout = "HIM_BRIDGE protocol=0 source_sha256=" + self.digest
        with self.assertRaisesRegex(ValueError, "version"):
            bridge.verify_bridge(self.binary, self.root, self.run)


if __name__ == "__main__":
    unittest.main()
