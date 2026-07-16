from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

FLEET_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FLEET_DIR))

from gpufleet import api
from gpufleet import __main__ as cli


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


def admission_args(**changes) -> argparse.Namespace:
    values = {
        "pod": "pod-test",
        "gpu": "a40",
        "name_prefix": "stwo-direct-bg-a40-",
        "max_usd_hr": 0.50,
        "min_vcpu": 16,
        "min_mem_gb": 62,
        "ttl_hours": 1.5,
        "idle_min": 15,
        "ready_timeout": 600,
        "one_shot": False,
        "failure_action": "stop",
        "purpose": "test",
    }
    values.update(changes)
    return argparse.Namespace(**values)


def record(events: list[str], name: str, result=None):
    def called(*_args, **_kwargs):
        events.append(name)
        return result

    return called


class ResumeApiTests(unittest.TestCase):
    def test_resume_is_one_attempt_and_requires_exact_returned_id(self) -> None:
        with mock.patch.object(
            api,
            "gql",
            return_value={"podResume": {"id": "wrong", "desiredStatus": "RUNNING"}},
        ) as gql:
            with self.assertRaisesRegex(api.ApiError, "wrong pod"):
                api.resume_pod("pod-test")
        self.assertEqual(gql.call_count, 1)
        self.assertEqual(gql.call_args.kwargs["retries"], 1)


class DeadmanTests(unittest.TestCase):
    def test_install_and_liveness_check_are_both_required(self) -> None:
        endpoint = cli.Endpoint("127.0.0.1", 22)
        log = Path("bootstrap.log")
        with (
            mock.patch.object(cli, "deadman_script", return_value="install-deadman"),
            mock.patch.object(cli, "ssh_run", side_effect=[0, 0]) as ssh,
        ):
            cli._install_deadman_first(
                endpoint,
                ttl_hours=1.5,
                idle_min=15,
                log_file=log,
            )
        self.assertEqual(ssh.call_count, 2)
        self.assertEqual(ssh.call_args_list[0].args, (endpoint, "install-deadman"))
        self.assertEqual(ssh.call_args_list[0].kwargs, {"timeout": 60, "log_file": log})
        self.assertIn("kill -0", ssh.call_args_list[1].args[1])
        self.assertEqual(ssh.call_args_list[1].kwargs, {"timeout": 20, "log_file": log})

    def test_install_or_liveness_failure_is_fatal(self) -> None:
        endpoint = cli.Endpoint("127.0.0.1", 22)
        for returns, message, expected_calls in (
            ([1], "installation failed", 1),
            ([0, 1], "liveness check failed", 2),
        ):
            with self.subTest(message=message):
                with (
                    mock.patch.object(cli, "deadman_script", return_value="install"),
                    mock.patch.object(cli, "ssh_run", side_effect=returns) as ssh,
                ):
                    with self.assertRaisesRegex(RuntimeError, message):
                        cli._install_deadman_first(
                            endpoint,
                            ttl_hours=1.5,
                            idle_min=15,
                            log_file=Path("bootstrap.log"),
                        )
                self.assertEqual(ssh.call_count, expected_calls)


class ExistingPodAdmissionTests(unittest.TestCase):
    def test_exited_pod_order_is_gate_validate_resume_deadman_bootstrap_gate(self) -> None:
        exited = pod(status="EXITED")
        ready = pod(status="RUNNING")
        events: list[str] = []
        with (
            mock.patch.object(cli, "_require_pregate", side_effect=record(events, "gate", True)),
            mock.patch.object(cli, "_bind_explicit_ssh_key", side_effect=record(events, "key")),
            mock.patch.object(
                cli.api,
                "secure_offer",
                side_effect=record(
                    events,
                    "offer",
                    {"display_name": ready.gpu, "usd_hr": 0.44},
                ),
            ),
            mock.patch.object(cli.api, "get_pod", side_effect=record(events, "get", exited)),
            mock.patch.object(cli.api, "resume_pod", side_effect=record(events, "resume", "RUNNING")),
            mock.patch.object(cli, "wait_ready", side_effect=record(events, "wait", ready)),
            mock.patch.object(cli, "_install_deadman_first", side_effect=record(events, "deadman")),
            mock.patch.object(cli, "bootstrap", side_effect=record(events, "bootstrap", True)),
            mock.patch.object(cli.pregate, "is_fresh", side_effect=record(events, "final-gate", True)),
            mock.patch.object(cli, "_sync_pods_conf", side_effect=record(events, "sync")),
            mock.patch.object(cli.ledger, "append"),
        ):
            admitted, _endpoint = cli._prepare_existing_pod(admission_args())
        self.assertEqual(admitted, ready)
        self.assertEqual(
            events,
            [
                "key", "gate", "offer", "get", "resume", "wait",
                "deadman", "bootstrap", "deadman", "final-gate", "sync",
            ],
        )

    def test_running_pod_is_adopted_without_resume_but_reinstalls_deadman(self) -> None:
        active = pod()
        events: list[str] = []
        with (
            mock.patch.object(cli, "_require_pregate", return_value=True),
            mock.patch.object(cli, "_bind_explicit_ssh_key"),
            mock.patch.object(
                cli.api,
                "secure_offer",
                return_value={"display_name": active.gpu, "usd_hr": 0.44},
            ),
            mock.patch.object(cli.api, "get_pod", return_value=active),
            mock.patch.object(cli.api, "resume_pod") as resume,
            mock.patch.object(cli, "wait_ready", return_value=active),
            mock.patch.object(cli, "_install_deadman_first", side_effect=record(events, "deadman")),
            mock.patch.object(cli, "bootstrap", side_effect=record(events, "bootstrap", True)),
            mock.patch.object(cli.pregate, "is_fresh", return_value=True),
            mock.patch.object(cli, "_sync_pods_conf"),
        ):
            cli._prepare_existing_pod(admission_args())
        resume.assert_not_called()
        self.assertEqual(events, ["deadman", "bootstrap", "deadman"])

    def test_stale_pregate_touches_no_provider_state(self) -> None:
        with (
            mock.patch.object(cli, "_bind_explicit_ssh_key"),
            mock.patch.object(cli, "_require_pregate", return_value=False),
            mock.patch.object(cli.api, "secure_offer") as offer,
            mock.patch.object(cli.api, "get_pod") as get,
            mock.patch.object(cli.api, "resume_pod") as resume,
        ):
            with self.assertRaisesRegex(RuntimeError, "fresh pregate"):
                cli._prepare_existing_pod(admission_args())
        offer.assert_not_called()
        get.assert_not_called()
        resume.assert_not_called()

    def test_current_price_ceiling_blocks_before_pod_read_or_resume(self) -> None:
        with (
            mock.patch.object(cli, "_require_pregate", return_value=True),
            mock.patch.object(cli, "_bind_explicit_ssh_key"),
            mock.patch.object(
                cli.api,
                "secure_offer",
                return_value={"display_name": "NVIDIA A40", "usd_hr": 0.51},
            ),
            mock.patch.object(cli.api, "get_pod") as get,
            mock.patch.object(cli.api, "resume_pod") as resume,
        ):
            with self.assertRaisesRegex(RuntimeError, "exceeds"):
                cli._prepare_existing_pod(admission_args())
        get.assert_not_called()
        resume.assert_not_called()

    def test_invalid_local_limits_block_before_pregate(self) -> None:
        for changes in (
            {"gpu": None},
            {"name_prefix": "unsafe_prefix-"},
            {"ttl_hours": math.nan},
            {"idle_min": 0},
            {"ready_timeout": math.inf},
            {"min_vcpu": 0},
            {"min_mem_gb": 0},
            {"max_usd_hr": 0},
        ):
            with self.subTest(changes=changes):
                with (
                    mock.patch.object(cli, "_bind_explicit_ssh_key"),
                    mock.patch.object(cli, "_require_pregate") as gate,
                ):
                    with self.assertRaises(ValueError):
                        cli._prepare_existing_pod(admission_args(**changes))
                gate.assert_not_called()

    def test_wrong_identity_shape_rate_or_state_blocks_mutation(self) -> None:
        valid = pod()
        cases = {
            "lookup id": replace(valid, id="other"),
            "unsafe id": replace(valid, id="unsafe id"),
            "name": replace(valid, name="somebody-elses-pod"),
            "gpu": replace(valid, gpu="NVIDIA RTX 4090"),
            "gpu bool": replace(valid, raw={"gpuCount": True}),
            "gpu count": replace(valid, raw={"gpuCount": 2}),
            "vcpu": replace(valid, vcpu=15),
            "memory": replace(valid, mem_gb=61),
            "zero rate": replace(valid, cost_per_hr=0),
            "nan rate": replace(valid, cost_per_hr=math.nan),
            "infinite rate": replace(valid, cost_per_hr=math.inf),
            "high rate": replace(valid, cost_per_hr=0.51),
            "state": replace(valid, status="PENDING"),
        }
        for name, candidate in cases.items():
            args = admission_args(pod=candidate.id) if name == "unsafe id" else admission_args()
            with self.subTest(name=name):
                with (
                    mock.patch.object(cli, "_require_pregate", return_value=True),
                    mock.patch.object(cli, "_bind_explicit_ssh_key"),
                    mock.patch.object(
                        cli.api,
                        "secure_offer",
                        return_value={"display_name": valid.gpu, "usd_hr": 0.44},
                    ),
                    mock.patch.object(cli.api, "get_pod", return_value=candidate),
                    mock.patch.object(cli.api, "resume_pod") as resume,
                    mock.patch.object(cli, "wait_ready") as wait,
                ):
                    with self.assertRaises(RuntimeError):
                        cli._prepare_existing_pod(args)
                resume.assert_not_called()
                wait.assert_not_called()

    def test_one_shot_never_reuses_an_exited_pod(self) -> None:
        exited = pod(status="EXITED")
        with (
            mock.patch.object(cli, "_require_pregate", return_value=True),
            mock.patch.object(cli, "_bind_explicit_ssh_key"),
            mock.patch.object(
                cli.api,
                "secure_offer",
                return_value={"display_name": exited.gpu, "usd_hr": 0.44},
            ),
            mock.patch.object(cli.api, "get_pod", return_value=exited),
            mock.patch.object(cli.api, "resume_pod") as resume,
        ):
            with self.assertRaisesRegex(RuntimeError, "cannot reuse"):
                cli._prepare_existing_pod(
                    admission_args(one_shot=True, failure_action="terminate")
                )
        resume.assert_not_called()

    def test_one_shot_requires_termination_policy_before_mutation(self) -> None:
        active = pod()
        with (
            mock.patch.object(cli, "_require_pregate", return_value=True),
            mock.patch.object(cli, "_bind_explicit_ssh_key"),
            mock.patch.object(
                cli.api,
                "secure_offer",
                return_value={"display_name": active.gpu, "usd_hr": 0.44},
            ),
            mock.patch.object(cli.api, "get_pod", return_value=active),
            mock.patch.object(cli.api, "resume_pod") as resume,
            mock.patch.object(cli, "wait_ready") as wait,
            mock.patch.object(cli.api, "stop_pod") as stop,
            mock.patch.object(cli.api, "terminate_pod") as terminate,
        ):
            with self.assertRaisesRegex(ValueError, "requires.*terminate"):
                cli._prepare_existing_pod(admission_args(one_shot=True))
        resume.assert_not_called()
        wait.assert_not_called()
        stop.assert_not_called()
        terminate.assert_not_called()

    def test_each_post_admission_failure_confirms_stop(self) -> None:
        for stage in (
            "resume",
            "wait",
            "ready-state",
            "deadman",
            "bootstrap",
            "post-deadman",
            "final-gate",
            "sync",
        ):
            initial = pod(status="EXITED") if stage == "resume" else pod()
            with self.subTest(stage=stage):
                with (
                    mock.patch.object(cli, "_require_pregate", return_value=True),
                    mock.patch.object(cli, "_bind_explicit_ssh_key"),
                    mock.patch.object(
                        cli.api,
                        "secure_offer",
                        return_value={"display_name": initial.gpu, "usd_hr": 0.44},
                    ),
                    mock.patch.object(cli.api, "get_pod", return_value=initial),
                    mock.patch.object(
                        cli.api,
                        "resume_pod",
                        side_effect=RuntimeError("resume failed") if stage == "resume" else None,
                    ),
                    mock.patch.object(
                        cli,
                        "wait_ready",
                        return_value=(
                            pod(status="EXITED") if stage == "ready-state" else pod()
                        ),
                        side_effect=RuntimeError("wait failed") if stage == "wait" else None,
                    ),
                    mock.patch.object(
                        cli,
                        "_install_deadman_first",
                        side_effect=(
                            RuntimeError("deadman failed")
                            if stage == "deadman"
                            else [None, RuntimeError("post-bootstrap deadman failed")]
                            if stage == "post-deadman"
                            else None
                        ),
                    ),
                    mock.patch.object(
                        cli,
                        "bootstrap",
                        return_value=stage != "bootstrap",
                    ),
                    mock.patch.object(cli.pregate, "is_fresh", return_value=stage != "final-gate"),
                    mock.patch.object(
                        cli,
                        "_sync_pods_conf",
                        side_effect=RuntimeError("sync failed") if stage == "sync" else None,
                    ),
                    mock.patch.object(cli.api, "stop_pod", return_value="EXITED") as stop,
                    mock.patch.object(cli.api, "terminate_pod") as terminate,
                    mock.patch.object(cli.ledger, "append"),
                ):
                    with self.assertRaises(RuntimeError):
                        cli._prepare_existing_pod(admission_args())
                stop.assert_called_once_with(initial.id)
                terminate.assert_not_called()

    def test_wrong_ready_id_still_cleans_up_the_admitted_pod(self) -> None:
        admitted = pod()
        wrong = replace(admitted, id="other-pod")
        with (
            mock.patch.object(cli, "_require_pregate", return_value=True),
            mock.patch.object(cli, "_bind_explicit_ssh_key"),
            mock.patch.object(
                cli.api,
                "secure_offer",
                return_value={"display_name": admitted.gpu, "usd_hr": 0.44},
            ),
            mock.patch.object(cli.api, "get_pod", return_value=admitted),
            mock.patch.object(cli, "wait_ready", return_value=wrong),
            mock.patch.object(cli.api, "stop_pod", return_value="EXITED") as stop,
            mock.patch.object(cli.ledger, "append"),
        ):
            with self.assertRaisesRegex(RuntimeError, "ready pod id changed"):
                cli._prepare_existing_pod(admission_args())
        stop.assert_called_once_with(admitted.id)

    def test_one_shot_failure_confirms_termination(self) -> None:
        active = pod()
        with (
            mock.patch.object(cli, "_require_pregate", return_value=True),
            mock.patch.object(cli, "_bind_explicit_ssh_key"),
            mock.patch.object(
                cli.api,
                "secure_offer",
                return_value={"display_name": active.gpu, "usd_hr": 0.44},
            ),
            mock.patch.object(cli.api, "get_pod", return_value=active),
            mock.patch.object(cli, "wait_ready", side_effect=RuntimeError("not ready")),
            mock.patch.object(cli.api, "stop_pod") as stop,
            mock.patch.object(cli.api, "terminate_pod") as terminate,
            mock.patch.object(cli.ledger, "append"),
        ):
            with self.assertRaisesRegex(RuntimeError, "not ready"):
                cli._prepare_existing_pod(
                    admission_args(one_shot=True, failure_action="terminate")
                )
        terminate.assert_called_once_with(active.id)
        stop.assert_not_called()

    def test_unconfirmed_cleanup_is_escalated_as_urgent(self) -> None:
        active = pod()
        with (
            mock.patch.object(cli, "_require_pregate", return_value=True),
            mock.patch.object(cli, "_bind_explicit_ssh_key"),
            mock.patch.object(
                cli.api,
                "secure_offer",
                return_value={"display_name": active.gpu, "usd_hr": 0.44},
            ),
            mock.patch.object(cli.api, "get_pod", return_value=active),
            mock.patch.object(cli, "wait_ready", side_effect=RuntimeError("not ready")),
            mock.patch.object(cli.api, "stop_pod", side_effect=RuntimeError("still running")),
        ):
            with self.assertRaisesRegex(RuntimeError, "URGENT.*not confirmed"):
                cli._prepare_existing_pod(admission_args())

    def test_explicit_ssh_identity_rejects_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "key"
            link = Path(directory) / "key-link"
            target.write_text("private key")
            link.symlink_to(target)
            with (
                mock.patch.dict(os.environ, {"RUNPOD_SSH_KEY": str(link)}),
                mock.patch.object(cli.pod_control, "SSH_OPTS", ["-o", "BatchMode=yes"]),
            ):
                with self.assertRaisesRegex(ValueError, "regular private-key"):
                    cli._bind_explicit_ssh_key()


class PodRunStaticTests(unittest.TestCase):
    def test_driver_has_no_raw_start_and_owns_only_after_guarded_resume(self) -> None:
        script = (FLEET_DIR.parent / "loop/pod_run.sh").read_text()
        self.assertNotIn("runpodctl pod start", script)
        resume = script.index('"$FLEET_CTL" "${RESUME_ARGS[@]}"')
        owned = script.index("POD_LIFECYCLE_OWNED=1", resume)
        endpoint = script.index("runpodctl ssh info", owned)
        self.assertLess(resume, owned)
        self.assertLess(owned, endpoint)

    def test_driver_passes_every_recipe_lease_constraint_to_resume(self) -> None:
        script = (FLEET_DIR.parent / "loop/pod_run.sh").read_text()
        required = (
            'resume --pod "$POD_ID" --gpu "$LEASE_GPU"',
            '--name-prefix "$LEASE_NAME_PREFIX"',
            '--max-usd-hr "$LEASE_MAX_USD_HR"',
            '--min-vcpu "$LEASE_MIN_VCPU" --min-mem-gb "$LEASE_MIN_MEM_GB"',
            '--ttl-hours "$LEASE_TTL_HOURS" --idle-min "$LEASE_IDLE_MIN"',
            '--failure-action "$POD_RUN_FINAL_ACTION"',
            '[[ "$LEASE_ONE_SHOT" == 1 ]] && RESUME_ARGS+=(--one-shot)',
            "one-shot recipes require an explicit BENCH_POD_ID",
        )
        for fragment in required:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, script)


if __name__ == "__main__":
    unittest.main()
