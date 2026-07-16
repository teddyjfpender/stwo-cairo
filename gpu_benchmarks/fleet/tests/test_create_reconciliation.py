from __future__ import annotations

import sys
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest import mock

FLEET_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FLEET_DIR))

from gpufleet import api
from gpufleet import __main__ as cli


def pod(identifier: str, name: str = "lease") -> api.PodInfo:
    return api.PodInfo(
        id=identifier,
        name=name,
        status="RUNNING",
        cost_per_hr=0.44,
        gpu="NVIDIA A40",
        dc="EU-1",
        vcpu=16,
        mem_gb=62,
        raw={"gpuCount": 1},
    )


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@contextmanager
def fake_reconciliation(clock: FakeClock):
    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(cli, "CREATE_RECONCILE_DISCOVERY_S", 4.0))
        stack.enter_context(mock.patch.object(cli, "CREATE_RECONCILE_TIMEOUT_S", 8.0))
        stack.enter_context(mock.patch.object(cli, "CREATE_RECONCILE_POLL_S", 1.0))
        stack.enter_context(
            mock.patch.object(cli.time, "monotonic", side_effect=clock.monotonic)
        )
        stack.enter_context(mock.patch.object(cli.time, "sleep", side_effect=clock.sleep))
        stack.enter_context(mock.patch.object(cli.ledger, "append"))
        yield


class CreateReconciliationTests(unittest.TestCase):

    def test_delayed_create_after_initial_empty_reads_is_terminated(self) -> None:
        candidate = pod("late")
        clock = FakeClock()
        with (
            fake_reconciliation(clock),
            mock.patch.object(
                cli.api,
                "list_pods",
                side_effect=[[], [], [candidate], [], []],
            ),
            mock.patch.object(cli.api, "terminate_pod") as terminate,
        ):
            cli._cleanup_failed_up("lease", None)
        terminate.assert_called_once_with(candidate.id)

    def test_every_exact_name_candidate_is_terminated_but_prefix_match_is_not(self) -> None:
        first = pod("first")
        second = pod("second")
        prefix_only = pod("other", name="lease-extra")
        clock = FakeClock()
        with (
            fake_reconciliation(clock),
            mock.patch.object(
                cli.api,
                "list_pods",
                side_effect=[[first, prefix_only, second], [], []],
            ),
            mock.patch.object(cli.api, "terminate_pod") as terminate,
        ):
            cli._cleanup_failed_up("lease", None)
        self.assertEqual(
            [call.args[0] for call in terminate.call_args_list],
            [first.id, second.id],
        )

    def test_stale_reappearance_is_terminated_again_before_stable_absence(self) -> None:
        candidate = pod("stale")
        clock = FakeClock()
        with (
            fake_reconciliation(clock),
            mock.patch.object(
                cli.api,
                "list_pods",
                side_effect=[[candidate], [], [candidate], [], []],
            ),
            mock.patch.object(cli.api, "terminate_pod") as terminate,
        ):
            cli._cleanup_failed_up("lease", None)
        self.assertEqual(terminate.call_count, 2)

    def test_transient_read_and_terminate_errors_are_retried(self) -> None:
        candidate = pod("retry")
        clock = FakeClock()
        with (
            fake_reconciliation(clock),
            mock.patch.object(
                cli.api,
                "list_pods",
                side_effect=[RuntimeError("read"), [candidate], [candidate], [], []],
            ),
            mock.patch.object(
                cli.api,
                "terminate_pod",
                side_effect=[RuntimeError("terminate"), None],
            ) as terminate,
        ):
            cli._cleanup_failed_up("lease", None)
        self.assertEqual(terminate.call_count, 2)

    def test_returned_candidate_is_terminated_even_when_roster_is_empty(self) -> None:
        returned = pod("returned")
        clock = FakeClock()
        with (
            fake_reconciliation(clock),
            mock.patch.object(cli.api, "list_pods", side_effect=[[], [], []]),
            mock.patch.object(cli.api, "terminate_pod") as terminate,
        ):
            cli._cleanup_failed_up("lease", returned)
        terminate.assert_called_once_with(returned.id)

    def test_no_observed_create_waits_the_full_discovery_window(self) -> None:
        clock = FakeClock()
        with (
            fake_reconciliation(clock),
            mock.patch.object(cli.api, "list_pods", return_value=[]),
            mock.patch.object(cli.api, "terminate_pod") as terminate,
        ):
            cli._cleanup_failed_up("lease", None)
        self.assertGreaterEqual(clock.now, 4.0)
        terminate.assert_not_called()

    def test_perpetual_stale_presence_fails_at_the_hard_deadline(self) -> None:
        candidate = pod("stuck")
        clock = FakeClock()
        with (
            fake_reconciliation(clock),
            mock.patch.object(cli, "CREATE_RECONCILE_TIMEOUT_S", 3.0),
            mock.patch.object(cli.api, "list_pods", return_value=[candidate]),
            mock.patch.object(cli.api, "terminate_pod") as terminate,
        ):
            with self.assertRaisesRegex(RuntimeError, "absence did not converge"):
                cli._cleanup_failed_up("lease", None)
        self.assertGreaterEqual(clock.now, 3.0)
        self.assertGreaterEqual(terminate.call_count, 3)


if __name__ == "__main__":
    unittest.main()
