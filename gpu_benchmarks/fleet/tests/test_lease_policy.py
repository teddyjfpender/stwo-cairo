from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

FLEET_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FLEET_DIR))

from gpufleet.lease_policy import PREFIX, load


BASE = {
    "one_shot": "true",
    "final_action": "terminate",
    "gpu": "a40",
    "gpu_count": "1",
    "min_vcpu": "16",
    "min_mem_gb": "62",
    "max_usd_hr": "0.50",
    "name_prefix": "stwo-direct-bg-a40-",
    "ttl_hours": "1.5",
    "idle_min": "15",
}


def line(values: dict[str, str] = BASE) -> str:
    return PREFIX + " ".join(f"{name}={value}" for name, value in values.items())


class LeasePolicyTests(unittest.TestCase):
    def load_text(self, text: str):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipe.phases"
            path.write_text(text)
            return load(path)

    def test_direct_a40_recipe_is_the_exact_one_shot_contract(self) -> None:
        recipe = FLEET_DIR.parent / "loop/recipes/direct_blake_g_native.phases"
        policy = load(recipe)
        self.assertEqual(policy.gpu, "a40")
        self.assertEqual(policy.gpu_count, 1)
        self.assertEqual((policy.min_vcpu, policy.min_mem_gb), (16, 62))
        self.assertEqual(policy.max_usd_hr, 0.5)
        self.assertEqual(policy.name_prefix, "stwo-direct-bg-a40-")
        self.assertEqual((policy.ttl_hours, policy.idle_min), (1.5, 15))
        self.assertTrue(policy.one_shot)
        self.assertEqual(policy.final_action, "terminate")
        self.assertEqual(
            policy.env_lines(),
            [
                "ONE_SHOT=1",
                "FINAL_ACTION=terminate",
                "GPU=a40",
                "GPU_COUNT=1",
                "MIN_VCPU=16",
                "MIN_MEM_GB=62",
                "MAX_USD_HR=0.5",
                "NAME_PREFIX=stwo-direct-bg-a40-",
                "TTL_HOURS=1.5",
                "IDLE_MIN=15",
            ],
        )

    def test_rejects_missing_duplicate_unknown_and_malformed_fields(self) -> None:
        missing = dict(BASE)
        del missing["gpu"]
        empty = {**BASE, "gpu": ""}
        cases = {
            "no directive": "phase test true\n",
            "two directives": f"{line()}\n{line()}\n",
            "missing": line(missing),
            "duplicate": line() + " gpu=a40",
            "unknown": line() + " surprise=yes",
            "malformed": line() + " broken",
            "too many equals": line() + " broken=a=b",
            "empty": line(empty),
        }
        for name, contents in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    self.load_text(contents)

    def test_rejects_invalid_semantics_and_noncanonical_numbers(self) -> None:
        cases = {
            "one_shot": {"one_shot": "yes"},
            "action": {"final_action": "delete"},
            "shot/action mismatch": {
                "one_shot": "false",
                "final_action": "terminate",
            },
            "unsafe gpu": {"gpu": "a40;touch"},
            "unsafe prefix": {"name_prefix": "bad_prefix-"},
            "prefix missing dash": {"name_prefix": "safe"},
            "two gpus": {"gpu_count": "2"},
            "leading zero": {"min_vcpu": "016"},
            "nonnumeric int": {"min_vcpu": "sixteen"},
            "zero int": {"idle_min": "0"},
            "negative int": {"min_mem_gb": "-1"},
            "nan": {"max_usd_hr": "nan"},
            "nonnumeric float": {"max_usd_hr": "cheap"},
            "infinity": {"ttl_hours": "inf"},
            "zero float": {"max_usd_hr": "0"},
        }
        for name, changes in cases.items():
            values = {**BASE, **changes}
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    self.load_text(line(values))

    def test_accepts_reusable_stop_contract(self) -> None:
        policy = self.load_text(
            line({**BASE, "one_shot": "false", "final_action": "stop"})
        )
        self.assertFalse(policy.one_shot)
        self.assertEqual(policy.final_action, "stop")


if __name__ == "__main__":
    unittest.main()
