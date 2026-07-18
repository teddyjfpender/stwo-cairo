from __future__ import annotations

import argparse
import math
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

FLEET_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FLEET_DIR))
DIRECT_RECIPE = FLEET_DIR.parent / "loop/recipes/direct_blake_g_native.phases"

from gpufleet import api
from gpufleet import __main__ as cli
from gpufleet.podctl import Endpoint


def pod(**changes) -> api.PodInfo:
    values = {
        "id": "pod-test",
        "name": "stwo-direct-bg-a40-test",
        "status": "RUNNING",
        "cost_per_hr": 0.44,
        "gpu": "NVIDIA A40",
        "dc": "EU-1",
        "vcpu": 16,
        "mem_gb": 62,
        "ssh_host": "127.0.0.1",
        "ssh_port": 22,
        "raw": {"gpuCount": 1},
    }
    values.update(changes)
    return api.PodInfo(**values)


def up_args(**changes) -> argparse.Namespace:
    values = {
        "gpu": "a40",
        "name": "stwo-direct-bg-a40-test",
        "image": "image@sha256:" + "1" * 64,
        "cloud": "SECURE",
        "disk_gb": 80,
        "volume_gb": 120,
        "volume_id": None,
        "min_vcpu": 16,
        "min_mem_gb": 62,
        "max_usd_hr": 0.50,
        "ttl_hours": 1.5,
        "idle_min": 15,
        "ready_timeout": 600,
        "purpose": "test",
        "recipe": str(DIRECT_RECIPE),
    }
    values.update(changes)
    return argparse.Namespace(**values)


class ProviderTests(unittest.TestCase):
    def test_secure_offer_uses_exact_secure_price(self) -> None:
        payload = {
            "gpuTypes": [{
                "id": "NVIDIA A40",
                "displayName": "NVIDIA A40",
                "secureCloud": True,
                "securePrice": "0.44",
                "lowestPrice": {"uninterruptablePrice": 0.10},
            }]
        }
        with mock.patch.object(api, "gql", return_value=payload):
            self.assertEqual(api.secure_offer("NVIDIA A40")["usd_hr"], 0.44)

    def test_secure_offer_rejects_ambiguous_or_invalid_values(self) -> None:
        cases = [
            [],
            [{"id": "wrong", "secureCloud": True, "securePrice": 0.1}],
            [
                {"id": "gpu", "secureCloud": True, "securePrice": 0.1},
                {"id": "gpu", "secureCloud": True, "securePrice": 0.1},
            ],
            [{"id": "gpu", "secureCloud": False, "securePrice": 0.1}],
            [{"id": "gpu", "secureCloud": True}],
            [{"id": "gpu", "secureCloud": True, "securePrice": 0}],
            [{"id": "gpu", "secureCloud": True, "securePrice": math.nan}],
            [{"id": "gpu", "secureCloud": True, "securePrice": math.inf}],
        ]
        for offers in cases:
            with self.subTest(offers=offers):
                with mock.patch.object(api, "gql", return_value={"gpuTypes": offers}):
                    with self.assertRaises(api.ApiError):
                        api.secure_offer("gpu")

    def test_create_mutation_is_issued_once(self) -> None:
        raw = {
            "id": "pod-test", "name": "lease", "desiredStatus": "RUNNING",
            "costPerHr": 0.44, "gpuCount": 1, "vcpuCount": 16,
            "memoryInGb": 62, "machine": {"gpuDisplayName": "NVIDIA A40"},
        }
        with mock.patch.object(
            api, "gql", return_value={"podFindAndDeployOnDemand": raw}
        ) as gql:
            api.create_pod(name="lease", gpu_type_id="NVIDIA A40", image="image")
        self.assertEqual(gql.call_args.kwargs["retries"], 1)
        self.assertEqual(gql.call_count, 1)

    def test_stop_reconciles_ambiguous_response(self) -> None:
        exited = pod(status="EXITED")
        with (
            mock.patch.object(api, "gql", side_effect=api.ApiError("timeout")) as gql,
            mock.patch.object(api, "_observe_pod", side_effect=[(exited, exited)] * 2),
            mock.patch.object(api.time, "sleep"),
        ):
            self.assertEqual(api.stop_pod(exited.id, poll_s=0), "EXITED")
        self.assertEqual(gql.call_args.kwargs["retries"], 1)
        self.assertEqual(gql.call_count, 1)

    def test_stop_rejects_unconfirmed_running_state(self) -> None:
        running = pod()
        with (
            mock.patch.object(
                api, "gql", return_value={"podStop": {"id": running.id}}
            ),
            mock.patch.object(api, "_observe_pod", return_value=(running, running)),
        ):
            with self.assertRaisesRegex(RuntimeError, "stop not confirmed"):
                api.stop_pod(running.id, timeout_s=0)

    def test_terminate_reconciles_ambiguous_absence(self) -> None:
        with (
            mock.patch.object(api, "gql", side_effect=api.ApiError("timeout")) as gql,
            mock.patch.object(api, "_observe_pod", side_effect=[(None, None)] * 2),
            mock.patch.object(api.time, "sleep"),
        ):
            api.terminate_pod("pod-test", poll_s=0)
        self.assertEqual(gql.call_args.kwargs["retries"], 1)
        self.assertEqual(gql.call_count, 1)

    def test_terminate_rejects_still_present_pod(self) -> None:
        running = pod()
        with (
            mock.patch.object(api, "gql", return_value={}),
            mock.patch.object(api, "_observe_pod", return_value=(running, running)),
        ):
            with self.assertRaisesRegex(RuntimeError, "terminate not confirmed"):
                api.terminate_pod(running.id, timeout_s=0)


class FleetCliTests(unittest.TestCase):
    def setUp(self) -> None:
        identity = mock.patch.object(cli, "_bind_explicit_ssh_key")
        identity.start()
        self.addCleanup(identity.stop)

    def test_require_pregate_is_freshness_only(self) -> None:
        with mock.patch.object(cli.pregate, "is_fresh", return_value=True):
            self.assertEqual(cli.cmd_require_pregate(None), 0)
        with (
            mock.patch.object(cli.pregate, "is_fresh", return_value=False),
            mock.patch.object(cli.pregate, "run") as run,
        ):
            self.assertEqual(cli.cmd_require_pregate(None), 1)
        run.assert_not_called()

    def test_wrapper_is_cwd_independent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            proc = subprocess.run(
                [str(FLEET_DIR / "gpufleet.sh"), "--help"],
                cwd=directory,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_legacy_raw_create_is_retired(self) -> None:
        script = FLEET_DIR / "pod_provision.sh"
        proc = subprocess.run(
            [str(script), "create", "--gpu", "NVIDIA A40"],
            capture_output=True,
            text=True,
            check=False,
        )
        output = proc.stdout + proc.stderr
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("create is retired", output)
        self.assertIn("gpufleet.sh up --recipe", output)
        self.assertNotIn("runpodctl pod create", output)

    def test_pod_run_uses_guarded_resume_after_trap(self) -> None:
        script = (FLEET_DIR.parent / "loop/pod_run.sh").read_text()
        trap = script.index("trap cleanup EXIT")
        resume = script.index('"$FLEET_CTL" "${RESUME_ARGS[@]}"')
        self.assertLess(trap, resume)
        self.assertNotIn('runpodctl pod start "$POD_ID"', script)

    def test_created_pod_validation_covers_shape_and_cost(self) -> None:
        args = up_args()
        offer = {"display_name": "NVIDIA A40"}
        cli._validate_created_pod(pod(), args.name, offer, args)
        invalid = [
            replace(pod(), id="unsafe id"),
            replace(pod(), name="wrong"),
            replace(pod(), gpu="wrong"),
            replace(pod(), raw={"gpuCount": 2}),
            replace(pod(), vcpu=15),
            replace(pod(), mem_gb=61),
            replace(pod(), cost_per_hr=0),
            replace(pod(), cost_per_hr=math.nan),
            replace(pod(), cost_per_hr=math.inf),
            replace(pod(), cost_per_hr=0.51),
        ]
        for candidate in invalid:
            with self.subTest(candidate=candidate):
                with self.assertRaises(RuntimeError):
                    cli._validate_created_pod(candidate, args.name, offer, args)

    def test_failed_pregate_blocks_up_before_offer(self) -> None:
        with (
            mock.patch.object(cli.pregate, "is_fresh", return_value=False),
            mock.patch.object(cli.pregate, "run", return_value=False),
            mock.patch.object(cli.api, "secure_offer") as offer,
        ):
            self.assertIsNone(cli.do_up(up_args()))
        offer.assert_not_called()

    def test_missing_recipe_blocks_up_before_pregate(self) -> None:
        with mock.patch.object(cli, "_require_pregate") as gate:
            with self.assertRaisesRegex(ValueError, "--recipe is required"):
                cli.do_up(up_args(recipe=None))
        gate.assert_not_called()

    def test_failed_pregate_blocks_run_before_pod_read(self) -> None:
        with (
            mock.patch.object(cli, "_require_pregate", return_value=False),
            mock.patch.object(cli, "_prepare_existing_pod") as prepare,
        ):
            self.assertEqual(cli.cmd_run(self._run_args()), 1)
        prepare.assert_not_called()

    def test_duplicate_name_blocks_create(self) -> None:
        existing = pod()
        with (
            mock.patch.object(cli, "_require_pregate", return_value=True),
            mock.patch.object(
                cli.api, "secure_offer",
                return_value={"display_name": existing.gpu, "usd_hr": 0.44},
            ),
            mock.patch.object(cli.api, "list_pods", return_value=[existing]),
            mock.patch.object(cli.api, "create_pod") as create,
        ):
            with self.assertRaisesRegex(RuntimeError, "duplicate lease name"):
                cli.do_up(up_args())
        create.assert_not_called()

    def test_bootstrap_failure_terminates_and_proves_name_absent(self) -> None:
        created = pod()
        lists = [[], [created], [created], [created], []]
        with (
            mock.patch.object(cli, "_require_pregate", return_value=True),
            mock.patch.object(
                cli.api, "secure_offer",
                return_value={"display_name": created.gpu, "usd_hr": 0.44},
            ),
            mock.patch.object(cli.api, "list_pods", side_effect=lists),
            mock.patch.object(cli.api, "create_pod", return_value=created),
            mock.patch.object(cli, "wait_ready", return_value=created),
            mock.patch.object(cli, "_install_deadman_first"),
            mock.patch.object(cli, "bootstrap", return_value=False),
            mock.patch.object(cli, "_cleanup_failed_up") as cleanup,
            mock.patch.object(cli.ledger, "append"),
        ):
            with self.assertRaisesRegex(RuntimeError, "bootstrap failed"):
                cli.do_up(up_args())
        cleanup.assert_called_once_with(created.name, created)

    def test_ambiguous_create_is_reconciled_by_exact_name(self) -> None:
        accepted = pod()
        lists = [[], [accepted], []]
        with (
            mock.patch.object(cli, "_require_pregate", return_value=True),
            mock.patch.object(
                cli.api, "secure_offer",
                return_value={"display_name": accepted.gpu, "usd_hr": 0.44},
            ),
            mock.patch.object(cli.api, "list_pods", side_effect=lists),
            mock.patch.object(cli.api, "create_pod", side_effect=api.ApiError("timeout")),
            mock.patch.object(cli, "_cleanup_failed_up") as cleanup,
            mock.patch.object(cli.ledger, "append"),
        ):
            with self.assertRaisesRegex(api.ApiError, "timeout"):
                cli.do_up(up_args())
        cleanup.assert_called_once_with(accepted.name, None)

    def test_interrupted_wait_still_terminates_returned_pod(self) -> None:
        created = pod()
        lists = [[], [created], [created], []]
        with (
            mock.patch.object(cli, "_require_pregate", return_value=True),
            mock.patch.object(
                cli.api, "secure_offer",
                return_value={"display_name": created.gpu, "usd_hr": 0.44},
            ),
            mock.patch.object(cli.api, "list_pods", side_effect=lists),
            mock.patch.object(cli.api, "create_pod", return_value=created),
            mock.patch.object(cli, "wait_ready", side_effect=KeyboardInterrupt),
            mock.patch.object(cli, "_cleanup_failed_up") as cleanup,
            mock.patch.object(cli.ledger, "append"),
        ):
            with self.assertRaises(KeyboardInterrupt):
                cli.do_up(up_args())
        cleanup.assert_called_once_with(created.name, created)

    def test_successful_up_rechecks_deadman_after_bootstrap(self) -> None:
        created = pod()
        with (
            mock.patch.object(cli, "_require_pregate", return_value=True),
            mock.patch.object(
                cli.api, "secure_offer",
                return_value={"display_name": created.gpu, "usd_hr": 0.44},
            ),
            mock.patch.object(
                cli.api, "list_pods", side_effect=[[], [created], [created]]
            ),
            mock.patch.object(cli.api, "create_pod", return_value=created),
            mock.patch.object(cli, "wait_ready", return_value=created),
            mock.patch.object(cli, "_install_deadman_first") as deadman,
            mock.patch.object(cli, "bootstrap", return_value=True),
            mock.patch.object(cli, "health_check", return_value={}),
            mock.patch.object(cli, "_sync_pods_conf"),
            mock.patch.object(cli.ledger, "append"),
        ):
            self.assertEqual(cli.do_up(up_args()), created)
        self.assertEqual(deadman.call_count, 2)

    def test_sn2_5mhz_up_uses_bounded_admission_and_rechecks_after_bootstrap(
        self,
    ) -> None:
        recipe = cli.STWO_CAIRO / cli.pregate.SN2_5MHZ_CHEAP_RECIPE
        created = pod(name="stwo-sn2-5mhz-ab-a40-deadbeef")
        args = up_args(
            recipe=str(recipe),
            name=created.name,
            volume_id=None,
        )
        receipt = {"scope": cli.pregate.SN2_5MHZ_CHEAP_SCOPE}
        with (
            mock.patch.object(
                cli.pregate, "admit_sn2_5mhz_cheap", return_value=receipt
            ) as cheap,
            mock.patch.object(cli, "_require_pregate") as formal,
            mock.patch.object(
                cli.api,
                "secure_offer",
                return_value={"display_name": created.gpu, "usd_hr": 0.44},
            ),
            mock.patch.object(
                cli.api, "list_pods", side_effect=[[], [created], [created]]
            ),
            mock.patch.object(cli.api, "create_pod", return_value=created),
            mock.patch.object(cli, "wait_ready", return_value=created),
            mock.patch.object(cli, "_install_deadman_first"),
            mock.patch.object(cli, "bootstrap", return_value=True),
            mock.patch.object(cli, "health_check", return_value={}),
            mock.patch.object(
                cli.pregate, "sn2_5mhz_cheap_is_current", return_value=True
            ) as current,
            mock.patch.object(cli, "_sync_pods_conf"),
            mock.patch.object(cli.ledger, "append"),
        ):
            self.assertEqual(cli.do_up(args), created)
        cheap.assert_called_once_with(recipe, cli.STWO, cli.STWO_CAIRO)
        current.assert_called_once_with(receipt, recipe, cli.STWO, cli.STWO_CAIRO)
        formal.assert_not_called()

    def test_sn2_5mhz_up_drift_after_bootstrap_terminates_created_pod(self) -> None:
        recipe = cli.STWO_CAIRO / cli.pregate.SN2_5MHZ_CHEAP_RECIPE
        created = pod(name="stwo-sn2-5mhz-ab-a40-deadbeef")
        args = up_args(recipe=str(recipe), name=created.name, volume_id=None)
        receipt = {"scope": cli.pregate.SN2_5MHZ_CHEAP_SCOPE}
        with (
            mock.patch.object(
                cli.pregate, "admit_sn2_5mhz_cheap", return_value=receipt
            ),
            mock.patch.object(
                cli.api,
                "secure_offer",
                return_value={"display_name": created.gpu, "usd_hr": 0.44},
            ),
            mock.patch.object(
                cli.api, "list_pods", side_effect=[[], [created], [created]]
            ),
            mock.patch.object(cli.api, "create_pod", return_value=created),
            mock.patch.object(cli, "wait_ready", return_value=created),
            mock.patch.object(cli, "_install_deadman_first"),
            mock.patch.object(cli, "bootstrap", return_value=True),
            mock.patch.object(cli, "health_check", return_value={}),
            mock.patch.object(
                cli.pregate, "sn2_5mhz_cheap_is_current", return_value=False
            ),
            mock.patch.object(cli, "_cleanup_failed_up") as cleanup,
            mock.patch.object(cli.ledger, "append"),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "source-bound provider admission changed"
            ):
                cli.do_up(args)
        cleanup.assert_called_once_with(created.name, created)

    def test_wrong_ready_id_rejects_and_cleans_the_created_pod(self) -> None:
        created = pod()
        wrong = replace(created, id="other-pod")
        lists = [[], [created], [wrong]]
        with (
            mock.patch.object(cli, "_require_pregate", return_value=True),
            mock.patch.object(
                cli.api, "secure_offer",
                return_value={"display_name": created.gpu, "usd_hr": 0.44},
            ),
            mock.patch.object(cli.api, "list_pods", side_effect=lists),
            mock.patch.object(cli.api, "create_pod", return_value=created),
            mock.patch.object(cli, "wait_ready", return_value=wrong),
            mock.patch.object(cli, "_cleanup_failed_up") as cleanup,
            mock.patch.object(cli.ledger, "append"),
        ):
            with self.assertRaisesRegex(RuntimeError, "ready pod id changed"):
                cli.do_up(up_args())
        cleanup.assert_called_once_with(created.name, created)

    def _run_args(self, **changes) -> argparse.Namespace:
        values = {
            "pod": "pod-test", "auto": None, "push": False, "keep": False,
            "terminate": False, "max_usd_hr": 0.5, "ttl_hours": 1.5,
            "idle_min": 15, "ready_timeout": 600, "gpu": "a40",
            "name_prefix": "stwo-direct-", "min_vcpu": 16,
            "min_mem_gb": 62, "one_shot": False, "failure_action": "stop",
            "purpose": "test", "manifest": "test.toml",
        }
        values.update(changes)
        return argparse.Namespace(**values)

    def test_run_confirms_stop_on_success(self) -> None:
        active = pod()
        run = mock.Mock()
        run.dir = Path("results")
        run.execute.return_value = {"run_id": "run", "passed": True}
        with (
            mock.patch.object(cli, "_require_pregate", return_value=True),
            mock.patch.object(cli, "_prepare_existing_pod", return_value=(active, Endpoint.of(active))),
            mock.patch.object(cli, "_sync_pods_conf"),
            mock.patch.object(cli, "ManifestRun", return_value=run),
            mock.patch.object(cli.api, "stop_pod", return_value="EXITED") as stop,
            mock.patch.object(cli.ledger, "append"),
        ):
            self.assertEqual(cli.cmd_run(self._run_args()), 0)
        stop.assert_called_once_with(active.id)

    def test_run_exception_still_confirms_stop(self) -> None:
        active = pod()
        run = mock.Mock()
        run.execute.side_effect = RuntimeError("execution failed")
        with (
            mock.patch.object(cli, "_require_pregate", return_value=True),
            mock.patch.object(cli, "_prepare_existing_pod", return_value=(active, Endpoint.of(active))),
            mock.patch.object(cli, "_sync_pods_conf"),
            mock.patch.object(cli, "ManifestRun", return_value=run),
            mock.patch.object(cli.api, "stop_pod", return_value="EXITED") as stop,
        ):
            with self.assertRaisesRegex(RuntimeError, "execution failed"):
                cli.cmd_run(self._run_args())
        stop.assert_called_once_with(active.id)

    def test_run_push_failure_still_confirms_stop(self) -> None:
        active = pod()
        with (
            mock.patch.object(cli, "_require_pregate", return_value=True),
            mock.patch.object(cli, "_prepare_existing_pod", return_value=(active, Endpoint.of(active))),
            mock.patch.object(cli, "_sync_pods_conf"),
            mock.patch.object(cli, "cmd_push", return_value=1),
            mock.patch.object(cli.api, "stop_pod", return_value="EXITED") as stop,
            mock.patch.object(cli.ledger, "append"),
        ):
            self.assertEqual(cli.cmd_run(self._run_args(push=True)), 1)
        stop.assert_called_once_with(active.id)

    def test_finalizer_failure_turns_pass_into_failure(self) -> None:
        active = pod()
        run = mock.Mock()
        run.dir = Path("results")
        run.execute.return_value = {"run_id": "run", "passed": True}
        with (
            mock.patch.object(cli, "_require_pregate", return_value=True),
            mock.patch.object(cli, "_prepare_existing_pod", return_value=(active, Endpoint.of(active))),
            mock.patch.object(cli, "_sync_pods_conf"),
            mock.patch.object(cli, "ManifestRun", return_value=run),
            mock.patch.object(cli.api, "stop_pod", side_effect=RuntimeError("still running")),
            mock.patch.object(cli.ledger, "append"),
        ):
            with self.assertRaisesRegex(RuntimeError, "still running"):
                cli.cmd_run(self._run_args())

    def test_keep_does_not_survive_execution_exception(self) -> None:
        active = pod()
        run = mock.Mock()
        run.execute.side_effect = RuntimeError("execution failed")
        with (
            mock.patch.object(cli, "_require_pregate", return_value=True),
            mock.patch.object(cli, "_prepare_existing_pod", return_value=(active, Endpoint.of(active))),
            mock.patch.object(cli, "_sync_pods_conf"),
            mock.patch.object(cli, "ManifestRun", return_value=run),
            mock.patch.object(cli.api, "stop_pod", return_value="EXITED") as stop,
            mock.patch.object(cli.ledger, "append"),
        ):
            with self.assertRaisesRegex(RuntimeError, "execution failed"):
                cli.cmd_run(self._run_args(keep=True))
        stop.assert_called_once_with(active.id)

    def test_run_terminate_option_confirms_absence(self) -> None:
        active = pod()
        run = mock.Mock()
        run.dir = Path("results")
        run.execute.return_value = {"run_id": "run", "passed": True}
        with (
            mock.patch.object(cli, "_require_pregate", return_value=True),
            mock.patch.object(cli, "_prepare_existing_pod", return_value=(active, Endpoint.of(active))),
            mock.patch.object(cli, "_sync_pods_conf"),
            mock.patch.object(cli, "ManifestRun", return_value=run),
            mock.patch.object(cli.api, "terminate_pod") as terminate,
            mock.patch.object(cli.ledger, "append"),
        ):
            self.assertEqual(cli.cmd_run(self._run_args(terminate=True)), 0)
        terminate.assert_called_once_with(active.id)

    def test_successful_one_shot_run_confirms_absence(self) -> None:
        active = pod()
        run = mock.Mock()
        run.dir = Path("results")
        run.execute.return_value = {"run_id": "run", "passed": True}
        with (
            mock.patch.object(cli, "_require_pregate", return_value=True),
            mock.patch.object(cli, "_prepare_existing_pod", return_value=(active, Endpoint.of(active))),
            mock.patch.object(cli, "ManifestRun", return_value=run),
            mock.patch.object(cli.api, "terminate_pod") as terminate,
            mock.patch.object(cli.ledger, "append"),
        ):
            self.assertEqual(
                cli.cmd_run(
                    self._run_args(one_shot=True, failure_action="terminate")
                ),
                0,
            )
        terminate.assert_called_once_with(active.id)

    def test_invalid_run_lifecycle_policy_blocks_before_pregate(self) -> None:
        cases = [
            {"keep": True, "terminate": True},
            {"keep": True, "one_shot": True, "failure_action": "terminate"},
            {"one_shot": True, "failure_action": "stop"},
        ]
        for changes in cases:
            with self.subTest(changes=changes):
                with mock.patch.object(cli, "_require_pregate") as gate:
                    with self.assertRaises(ValueError):
                        cli.cmd_run(self._run_args(**changes))
                gate.assert_not_called()

    def test_run_auto_is_retired_before_pregate_or_create(self) -> None:
        with (
            mock.patch.object(cli, "_require_pregate") as gate,
            mock.patch.object(cli.api, "create_pod") as create,
        ):
            with self.assertRaisesRegex(ValueError, "--auto is retired"):
                cli.cmd_run(self._run_args(pod=None, auto="a40"))
        gate.assert_not_called()
        create.assert_not_called()

    def test_explicit_absent_terminate_is_still_confirmed(self) -> None:
        args = argparse.Namespace(all=False, pod="pod-test")
        with (
            mock.patch.object(cli.api, "get_pod", return_value=None),
            mock.patch.object(cli.api, "terminate_pod") as terminate,
        ):
            self.assertEqual(cli._lifecycle(args, "terminate"), 0)
        terminate.assert_called_once_with("pod-test")

    def test_all_lifecycle_attempts_every_target_before_failing(self) -> None:
        first = pod(id="pod-one")
        second = pod(id="pod-two")
        args = argparse.Namespace(all=True, pod=None)
        with (
            mock.patch.object(cli.api, "list_pods", return_value=[first, second]),
            mock.patch.object(
                cli.api,
                "terminate_pod",
                side_effect=[RuntimeError("ambiguous"), None],
            ) as terminate,
            mock.patch.object(cli.ledger, "append"),
        ):
            with self.assertRaisesRegex(RuntimeError, "pod-one: ambiguous"):
                cli._lifecycle(args, "terminate")
        self.assertEqual(
            [call.args[0] for call in terminate.call_args_list],
            ["pod-one", "pod-two"],
        )


if __name__ == "__main__":
    unittest.main()
