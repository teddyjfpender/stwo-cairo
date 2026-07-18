#!/usr/bin/env python3
"""Seal the fail-soft A40 candidate run with one fail-closed verdict."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import tempfile


HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HEX_U64 = re.compile(r"0x(?!0{16})[0-9a-f]{16}\Z")
TARGETS = {
    "relation": "prepared_relation_native",
    "quotient": "replacement_stage4_native",
    "composition": "prepared_composition_stripes_direct_native",
}
RELATION_CHECKS = {
    "same_fixture_host_bytes",
    "adaptive_eager_bytes",
    "baseline_eager_bytes",
    "adaptive_captured_mutation_bytes",
    "baseline_captured_mutation_bytes",
    "compact_mode_guard",
    "zero_denominator_selector_differential",
    "adaptive_zero_denominator_fail_closed",
    "baseline_zero_denominator_fail_closed",
    "invalid_input_guards",
    "loaded_resource_abi",
    "adaptive_resource_policy_admitted",
    "eager_positive_median_speedup",
    "captured_positive_median_speedup",
}
RELATION_PERFORMANCE_CHECKS = {
    "eager_positive_median_speedup",
    "captured_positive_median_speedup",
}
RELATION_CORRECTNESS_CHECKS = RELATION_CHECKS - RELATION_PERFORMANCE_CHECKS
QUOTIENT_CHECKS = {
    "exact_plan_receipt",
    "dense_native_independent_cpu_oracle",
    "eager_staged_prepacked_identity",
    "captured_replay_status_reset",
    "invalid_descriptor_rejects_stale_output",
    "replay_recovers_after_status_error",
    "source_and_guard_preservation",
}


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: pathlib.Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_json(path: pathlib.Path) -> object:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def cargo_artifact(cargo_json: pathlib.Path, target: str) -> pathlib.Path:
    matches: set[pathlib.Path] = set()
    with cargo_json.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            info = value.get("target", {})
            executable = value.get("executable")
            if (
                value.get("reason") == "compiler-artifact"
                and info.get("name") == target
                and "test" in info.get("kind", [])
                and isinstance(executable, str)
            ):
                matches.add(pathlib.Path(executable))
    if len(matches) != 1:
        raise ValueError(f"{target}: expected one test executable, got {sorted(map(str, matches))}")
    executable = matches.pop()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError(f"{target}: non-executable Cargo artifact {executable}")
    return executable.resolve()


def artifact_command(args: argparse.Namespace) -> int:
    executable = cargo_artifact(args.cargo_json, args.target)
    record = {
        "schema": "stwo.cheap-gpu.test-binary.v1",
        "target": args.target,
        "path": str(executable),
        "sha256": sha256(executable),
        "size_bytes": executable.stat().st_size,
    }
    atomic_json(args.out, record)
    print(executable)
    return 0


def output(command: list[str], *, cwd: pathlib.Path | None = None) -> str:
    return subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=True,
        cwd=cwd,
    ).stdout.strip()


def capture_command(args: argparse.Namespace) -> int:
    errors: list[str] = []
    query = (
        "name,uuid,pci.bus_id,compute_cap,memory.total,driver_version,"
        "power.limit,clocks.max.graphics,clocks.max.memory"
    )
    try:
        rows = list(
            csv.reader(
                output(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"])
                .splitlines()
            )
        )
    except (OSError, subprocess.CalledProcessError) as error:
        rows = []
        errors.append(f"nvidia-smi identity failed: {error}")
    if len(rows) != 1 or len(rows[0]) != 9:
        errors.append(f"expected one nine-field GPU identity row, got {rows!r}")
        gpu = {}
    else:
        fields = [field.strip() for field in rows[0]]
        gpu = dict(
            zip(
                (
                    "name",
                    "uuid",
                    "pci_bus_id",
                    "compute_capability",
                    "memory_total_mib",
                    "driver_version",
                    "power_limit_w",
                    "max_graphics_clock_mhz",
                    "max_memory_clock_mhz",
                ),
                fields,
                strict=True,
            )
        )
    tools: dict[str, str] = {}
    for name, command, cwd in (
        ("nvcc", ["nvcc", "--version"], None),
        ("stwo_rustc", ["rustc", "-Vv"], args.stwo),
        ("stwo_cargo", ["cargo", "-V"], args.stwo),
        ("stwo_cairo_rustc", ["rustc", "-Vv"], args.cairo),
        ("stwo_cairo_cargo", ["cargo", "-V"], args.cairo),
    ):
        try:
            tools[name] = output(command, cwd=cwd)
        except (OSError, subprocess.CalledProcessError) as error:
            errors.append(f"{name} identity failed: {error}")
    source = {
        "stwo_head": os.environ.get("STWO_PARITY_REF_STWO_HEAD", ""),
        "stwo_worktree_hash": os.environ.get("STWO_PARITY_REF_STWO_WORKTREE_HASH", ""),
        "stwo_cairo_head": os.environ.get("STWO_PARITY_REF_STWO_CAIRO_HEAD", ""),
        "stwo_cairo_worktree_hash": os.environ.get(
            "STWO_PARITY_REF_STWO_CAIRO_WORKTREE_HASH", ""
        ),
    }
    if not HEX40.fullmatch(source["stwo_head"]) or not HEX40.fullmatch(
        source["stwo_cairo_head"]
    ):
        errors.append("source heads are not exact lowercase Git identities")
    if not HEX64.fullmatch(source["stwo_worktree_hash"]) or not HEX64.fullmatch(
        source["stwo_cairo_worktree_hash"]
    ):
        errors.append("source projections are not exact lowercase SHA-256 identities")
    binaries: dict[str, object] = {}
    for label, target in TARGETS.items():
        path = args.run / f"{label}.binary.json"
        try:
            record = load_json(path)
            executable = pathlib.Path(record["path"])
            if (
                record.get("schema") != "stwo.cheap-gpu.test-binary.v1"
                or record.get("target") != target
                or not HEX64.fullmatch(str(record.get("sha256", "")))
                or sha256(executable) != record["sha256"]
            ):
                raise ValueError("artifact identity changed after execution")
            binaries[label] = record
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"{label} binary identity: {error}")
    try:
        manifest_sha = sha256(args.aot_manifest)
    except OSError as error:
        manifest_sha = ""
        errors.append(f"AOT manifest identity: {error}")
    record = {
        "schema": "stwo.sn2-5mhz.cheap-gpu-environment.v1",
        "passed": not errors,
        "errors": errors,
        "requested_cuda_arch": os.environ.get("STWO_CUDA_ARCH", ""),
        "source": source,
        "gpu": gpu,
        "toolchain": tools,
        "aot_manifest": {
            "path": str(args.aot_manifest),
            "sha256": manifest_sha,
        },
        "test_binaries": binaries,
    }
    atomic_json(args.out, record)
    print(json.dumps(record, sort_keys=True))
    return int(bool(errors))


class Checks:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            self.errors.append(message)

    def json(self, path: pathlib.Path, label: str) -> dict:
        try:
            value = load_json(path)
        except (OSError, json.JSONDecodeError) as error:
            self.errors.append(f"{label}: missing or malformed JSON: {error}")
            return {}
        if not isinstance(value, dict):
            self.errors.append(f"{label}: root must be an object")
            return {}
        return value


def positive(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def resource(checks: Checks, value: object, label: str, *, target_sm: int = 86) -> None:
    checks.require(isinstance(value, dict), f"{label}: resource facts missing")
    if not isinstance(value, dict):
        return
    checks.require(value.get("abi_version") == 1, f"{label}: ABI drift")
    checks.require(value.get("reserved") == 0, f"{label}: reserved field is nonzero")
    checks.require(value.get("max_threads_per_block", 0) >= 256, f"{label}: thread cap")
    checks.require(value.get("registers_per_thread", 0) > 0, f"{label}: registers missing")
    checks.require(value.get("binary_version") == target_sm, f"{label}: wrong loaded SM")
    checks.require(
        0 < value.get("ptx_version", 0) <= target_sm, f"{label}: invalid PTX version"
    )
    checks.require(value.get("local_bytes", -1) >= 0, f"{label}: local bytes missing")
    checks.require(
        value.get("static_shared_bytes", -1) >= 0, f"{label}: static shared bytes missing"
    )


def composition_resource(checks: Checks, value: object, label: str) -> None:
    checks.require(isinstance(value, dict), f"{label}: resource facts missing")
    if not isinstance(value, dict):
        return
    checks.require(value.get("max_threads_per_block", 0) >= 128, f"{label}: thread cap")
    checks.require(
        0 < value.get("registers_per_thread", 0) <= 128, f"{label}: register envelope"
    )
    checks.require(value.get("target_sm") == 86, f"{label}: target SM")
    checks.require(value.get("binary_version") == 86, f"{label}: wrong loaded SM")
    checks.require(0 < value.get("ptx_version", 0) <= 86, f"{label}: invalid PTX version")
    checks.require(value.get("local_bytes") == 0, f"{label}: local-memory spill")
    checks.require(value.get("static_shared_bytes") == 0, f"{label}: static shared memory")
    checks.require(value.get("dynamic_shared_bytes") == 0, f"{label}: dynamic shared memory")
    for dimension in ("grid", "block"):
        shape = value.get(dimension, [])
        checks.require(
            isinstance(shape, list) and len(shape) == 3 and all(item > 0 for item in shape),
            f"{label}: invalid {dimension}",
        )


def relation(checks: Checks, value: dict, stwo_head: str) -> dict:
    checks.require(value.get("schema") == "stwo.prepared-relation.same-binary-ab.v2", "relation schema")
    checks.require(value.get("passed") is True, "relation comparator did not pass")
    checks.require(value.get("git_commit") == stwo_head, "relation source head mismatch")
    checks.require(value.get("baseline_source_commit") == "0016f4b5", "relation baseline source")
    checks.require(value.get("baseline") == "pre_adaptive_columns_le_512_one_read", "relation baseline")
    checks.require(value.get("candidate") == "adaptive_tuple_width_gt_32_one_read", "relation candidate")
    checks.require(
        value.get("ordering") == "alternating_baseline_candidate_then_candidate_baseline",
        "relation timing order",
    )
    checks.require(value.get("percentiles") == "sorted_linear_index", "relation percentile rule")
    checks.require(
        HEX64.fullmatch(str(value.get("output_blake3", ""))) is not None,
        "relation output identity",
    )
    selector = value.get("selector_fixture", {})
    checks.require(selector.get("adaptive_lane_change_batches", 0) > 0, "relation selector did not change a lane")
    checks.require(selector.get("shared_one_read_batches", 0) > 0, "relation shared one-read lane")
    poison = value.get("zero_denominator_fixture", {})
    for field in ("batch_index", "instance_index", "column_index", "source_index"):
        checks.require(poison.get(field) == 0, f"relation poison {field} drift")
    checks.require(poison.get("tuple_class") == "at_most_32_words", "relation poison tuple class")
    checks.require(0 < poison.get("columns", 0) <= 512, "relation poison column count")
    checks.require(
        0 < poison.get("max_tuple_words", 0) <= 32, "relation poison maximum tuple"
    )
    checks.require(
        0 < poison.get("poisoned_use_tuple_words", 0) <= 32,
        "relation poisoned tuple width",
    )
    checks.require(poison.get("baseline_lane") == "one_read", "relation poison baseline lane")
    checks.require(poison.get("adaptive_lane") == "suffix_recompute", "relation poison candidate lane")
    named_checks = value.get("checks", {})
    checks.require(
        set(named_checks) == RELATION_CHECKS,
        "relation check set",
    )
    checks.require(
        all(named_checks.get(name) is True for name in RELATION_CORRECTNESS_CHECKS),
        "relation correctness checks",
    )
    loaded = value.get("loaded_functions", {})
    checks.require(
        set(loaded) == {"adaptive_relation_fused_kernel", "all_one_read_test_kernel"},
        "relation loaded-function set",
    )
    for name, facts in loaded.items():
        resource(checks, facts, f"relation {name}")
    static = value.get("static", {})
    checks.require(
        HEX64.fullmatch(str(static.get("source_identity", ""))) is not None,
        "relation static source identity",
    )
    checks.require(
        HEX64.fullmatch(str(static.get("module_build_identity", ""))) is not None,
        "relation module identity",
    )
    checks.require(86 in static.get("target_sms", []), "relation static module lacks sm_86")
    policy = value.get("adaptive_resource_policy", {})
    checks.require(policy.get("mode") == "first_characterization_report_only", "relation A40 resource policy")
    checks.require(policy.get("binary_version") == 86, "relation policy SM")
    checks.require(policy.get("ceiling_enforced") is False, "relation incorrectly applied SM90 ceiling")
    speedups: dict[str, float] = {}
    for mode in ("eager", "captured"):
        timing = value.get(mode, {})
        checks.require(timing.get("warmups") == 5, f"relation {mode} warmups")
        checks.require(timing.get("iterations") == 30, f"relation {mode} iterations")
        speedup = timing.get("candidate_speedup")
        checks.require(positive(speedup) and speedup > 1.0, f"relation {mode} speedup")
        checks.require(
            named_checks.get(f"{mode}_positive_median_speedup")
            is (positive(speedup) and speedup > 1.0),
            f"relation {mode} performance-check drift",
        )
        for arm in ("baseline", "candidate"):
            stats = timing.get(arm, {})
            checks.require(len(stats.get("samples_ms", [])) == 30, f"relation {mode} {arm} samples")
            for field in ("median_ms", "p10_ms", "p90_ms"):
                checks.require(positive(stats.get(field)), f"relation {mode} {arm} {field}")
        if positive(speedup):
            speedups[mode] = speedup
    return speedups


def quotient(checks: Checks, value: dict, stwo_head: str) -> dict:
    checks.require(value.get("schema") == "stwo.replacement-stage4-native.v1", "quotient schema")
    checks.require(value.get("passed") is True, "quotient admission did not pass")
    checks.require(value.get("git_commit") == stwo_head, "quotient source head mismatch")
    checks.require(value.get("fixture_filter") == "staged-prepacked-quotient", "quotient filter")
    checks.require(value.get("requested_cuda_arch") == "sm_86", "quotient requested SM")
    checks.require(value.get("performance_requested") is True, "quotient timing omitted")
    checks.require(value.get("performance_failure") is None, "quotient timing failure was swallowed")
    checks.require(HEX64.fullmatch(str(value.get("executable_blake3", ""))) is not None, "quotient executable identity")
    checks.require(HEX64.fullmatch(str(value.get("source_blake3", ""))) is not None, "quotient source identity")
    fixtures = value.get("fixtures", [])
    checks.require(len(fixtures) == 1, "quotient fixture count")
    if len(fixtures) == 1:
        fixture = fixtures[0]
        checks.require(fixture.get("name") == "staged-prepacked-quotient-boundary", "quotient fixture")
        checks.require(fixture.get("cases") == 4, "quotient fixture cases")
        fixture_checks = fixture.get("checks", {})
        checks.require(
            set(fixture_checks) == QUOTIENT_CHECKS and all(fixture_checks.values()),
            "quotient correctness-check set",
        )
    expected = {
        f"staged-prepacked-quotient-{mode}-log{log}"
        for log in (18, 20)
        for mode in ("eager", "captured")
    }
    performance = value.get("performance", [])
    checks.require({entry.get("name") for entry in performance} == expected, "quotient timing matrix")
    speedups: dict[str, float] = {}
    roles = {
        "baseline_staged_packed",
        "candidate_prepacked_prepare",
        "candidate_prepacked_validate",
        "candidate_prepacked_hot",
    }
    symbols = {
        "baseline_staged_packed": "stwo_quotient_numerator_packed_single_write_kernel",
        "candidate_prepacked_prepare": "stwo_prepare_quotient_numerator_prepacked_terms_kernel",
        "candidate_prepacked_validate": "stwo_validate_quotient_numerator_prepacked_terms_kernel",
        "candidate_prepacked_hot": "stwo_quotient_numerator_prepacked_single_write_kernel",
    }
    first_loaded: list[dict] | None = None
    for entry in performance:
        name = str(entry.get("name", "unknown"))
        parameters = entry.get("parameters", {})
        expected_log = 18 if name.endswith("log18") else 20 if name.endswith("log20") else None
        checks.require(parameters.get("lifting_log_size") == expected_log, f"{name}: lifting log")
        checks.require(parameters.get("groups") == 4, f"{name}: group count")
        checks.require(parameters.get("terms") == 20, f"{name}: term count")
        checks.require(parameters.get("prepacked_used_words") == 157, f"{name}: packed words")
        checks.require(parameters.get("stream_count") == 1, f"{name}: stream count")
        checks.require(parameters.get("source_buffer_sets") == 1, f"{name}: source sets")
        checks.require(
            parameters.get("candidate_status_observations") == 35,
            f"{name}: status fences",
        )
        arena = entry.get("arena_bytes", {})
        checks.require(
            set(arena) == {"shared_single_stream"} and arena["shared_single_stream"] > 0,
            f"{name}: shared arena",
        )
        traffic = entry.get("traffic_bytes", {})
        checks.require(
            traffic.get("candidate_status_fence_d2h_bytes_per_replay_outside_kernel_time")
            == 4,
            f"{name}: status fence bytes",
        )
        checks.require(
            traffic.get("candidate_status_fence_d2h_bytes_total_outside_kernel_time")
            == 140,
            f"{name}: total status fence bytes",
        )
        loaded = entry.get("loaded_functions", [])
        checks.require(
            len(loaded) == 4 and {facts.get("role") for facts in loaded} == roles,
            f"{name}: resources",
        )
        for facts in loaded:
            resource(checks, facts, f"{name} {facts.get('role')}")
            checks.require(
                facts.get("symbol") == symbols.get(facts.get("role")),
                f"{name}: loaded symbol",
            )
            checks.require(facts.get("launch_threads") == 256, f"{name}: launch threads")
            checks.require(
                facts.get("dynamic_shared_bytes") == 0, f"{name}: dynamic shared memory"
            )
        if first_loaded is None:
            first_loaded = loaded
        else:
            checks.require(loaded == first_loaded, f"{name}: loaded resources drifted")
        for arm in ("baseline", "candidate"):
            timing = entry.get(arm, {})
            checks.require(timing.get("warmups") == 5, f"{name}: {arm} warmups")
            checks.require(timing.get("iterations") == 30, f"{name}: {arm} iterations")
            for field in ("median_ms", "p10_ms", "p90_ms"):
                checks.require(positive(timing.get(field)), f"{name}: {arm} {field}")
        speedup = entry.get("speedup")
        checks.require(
            positive(speedup) and speedup > 1.0,
            f"{name}: candidate did not exceed baseline",
        )
        if positive(speedup):
            speedups[name] = speedup
    return speedups


def composition(checks: Checks, value: dict) -> dict:
    checks.require(value.get("schema") == "stwo.composition.stripe-direct-diagnostic.v1", "composition schema")
    checks.require(value.get("passed") is True, "composition comparator did not pass")
    checks.require(value.get("diagnostic_only") is True, "composition diagnostic label")
    checks.require(value.get("promotion_credit") is False, "composition improperly claims promotion credit")
    checks.require(value.get("real_sn2_coverage") is False, "composition improperly claims SN2 coverage")
    checks.require(value.get("target_useful_mhz") == 5.0, "composition target drift")
    named = value.get("correctness_checks", {})
    checks.require(named.get("multiple_evaluation_domains") == [7, 13, 24], "composition domain fixture")
    checks.require(named.get("runtime_strict_rejections") == 0, "composition strict AOT miss")
    for field in (
        "direct_split_output",
        "shared_source_inputs",
        "single_process_device_arena_context_main_stream",
        "all_retained_bytes_equal_eager",
        "all_retained_bytes_equal_mutated_capture_replay",
        "mutated_replay_digest_changed",
        "installed_identity_and_resources_present",
    ):
        checks.require(named.get(field) is True, f"composition {field}")
    eager_digest = str(value.get("eager_retained_blake3", ""))
    replay_digest = str(value.get("mutated_replay_retained_blake3", ""))
    checks.require(
        HEX64.fullmatch(eager_digest) is not None
        and HEX64.fullmatch(replay_digest) is not None
        and eager_digest != "0" * 64
        and replay_digest != "0" * 64
        and eager_digest != replay_digest,
        "composition mutation was not observable",
    )
    functions = value.get("installed_functions", [])
    checks.require(
        isinstance(value.get("stripe_count"), int)
        and value["stripe_count"] > 0
        and len(functions) == value["stripe_count"],
        "composition stripe/function count",
    )
    roles: set[tuple[object, object]] = set()
    for index, facts in enumerate(functions):
        composition_resource(checks, facts, f"composition function {index}")
        role = (facts.get("component"), facts.get("kernel"))
        checks.require(
            all(isinstance(item, int) and item >= 0 for item in role),
            f"composition function {index}: role",
        )
        checks.require(role not in roles, f"composition function {index}: duplicate role")
        roles.add(role)
        checks.require(
            HEX_U64.fullmatch(str(facts.get("cache_key", ""))) is not None,
            f"composition function {index}: cache key",
        )
        checks.require(
            HEX_U64.fullmatch(str(facts.get("semantic_hash", ""))) is not None,
            f"composition function {index}: semantic hash",
        )
        source_identity = str(facts.get("source_identity", ""))
        cubin_identity = str(facts.get("cubin_identity", ""))
        checks.require(
            HEX64.fullmatch(source_identity) is not None and source_identity != "0" * 64,
            f"composition function {index}: source identity",
        )
        checks.require(
            HEX64.fullmatch(cubin_identity) is not None and cubin_identity != "0" * 64,
            f"composition function {index}: cubin identity",
        )
    timing = value.get("timing", {})
    checks.require(timing.get("order") == "ABBA", "composition timing order")
    checks.require(timing.get("same_exec_context") is True, "composition execution context")
    checks.require(timing.get("promotion_credit") is False, "composition timing credit")
    observed: dict[str, float] = {}
    for mode in ("eager", "captured"):
        sample = timing.get(mode, {})
        checks.require(sample.get("samples_per_arm", 0) >= 2, f"composition {mode} samples")
        checks.require(positive(sample.get("wave_source_jit_median_ms")), f"composition {mode} wave median")
        checks.require(positive(sample.get("installed_stripes_median_ms")), f"composition {mode} stripe median")
        ratio = sample.get("observed_candidate_over_wave")
        checks.require(positive(ratio), f"composition {mode} ratio")
        if positive(ratio):
            observed[mode] = ratio
    return observed


def validate_run(run: pathlib.Path) -> dict:
    checks = Checks()
    for name in ("relation", "quotient", "composition", "environment"):
        try:
            raw = (run / f"{name}.raw.rc").read_text(encoding="ascii").strip()
        except OSError as error:
            raw = ""
            checks.errors.append(f"{name}: raw status missing: {error}")
        checks.require(raw == "0", f"{name}: raw command failed with {raw!r}")
    environment = checks.json(run / "environment.json", "environment")
    checks.require(environment.get("schema") == "stwo.sn2-5mhz.cheap-gpu-environment.v1", "environment schema")
    checks.require(environment.get("passed") is True, "environment identity incomplete")
    checks.require(environment.get("requested_cuda_arch") == "sm_86", "environment requested SM")
    gpu = environment.get("gpu", {})
    checks.require(gpu.get("compute_capability") == "8.6", "hardware is not one sm_86 A40")
    checks.require("A40" in str(gpu.get("name", "")), "hardware is not an A40")
    checks.require(str(gpu.get("uuid", "")).startswith("GPU-"), "hardware UUID missing")
    checks.require(bool(gpu.get("pci_bus_id")), "hardware PCI bus missing")
    checks.require(bool(gpu.get("driver_version")), "hardware driver missing")
    for field in ("memory_total_mib", "power_limit_w", "max_graphics_clock_mhz", "max_memory_clock_mhz"):
        try:
            checks.require(float(gpu.get(field, 0)) > 0, f"hardware {field} missing")
        except (TypeError, ValueError):
            checks.require(False, f"hardware {field} malformed")
    source = environment.get("source", {})
    checks.require(HEX40.fullmatch(str(source.get("stwo_head", ""))) is not None, "stwo head")
    checks.require(HEX40.fullmatch(str(source.get("stwo_cairo_head", ""))) is not None, "stwo-cairo head")
    checks.require(HEX64.fullmatch(str(source.get("stwo_worktree_hash", ""))) is not None, "stwo projection")
    checks.require(HEX64.fullmatch(str(source.get("stwo_cairo_worktree_hash", ""))) is not None, "stwo-cairo projection")
    expected_source = {
        "stwo_head": os.environ.get("STWO_PARITY_REF_STWO_HEAD", ""),
        "stwo_worktree_hash": os.environ.get("STWO_PARITY_REF_STWO_WORKTREE_HASH", ""),
        "stwo_cairo_head": os.environ.get("STWO_PARITY_REF_STWO_CAIRO_HEAD", ""),
        "stwo_cairo_worktree_hash": os.environ.get(
            "STWO_PARITY_REF_STWO_CAIRO_WORKTREE_HASH", ""
        ),
    }
    checks.require(source == expected_source, "environment source differs from pod projection")
    manifest = environment.get("aot_manifest", {})
    manifest_digest = str(manifest.get("sha256", ""))
    checks.require(HEX64.fullmatch(manifest_digest) is not None, "AOT manifest SHA-256")
    try:
        checks.require(
            sha256(pathlib.Path(manifest.get("path", ""))) == manifest_digest,
            "AOT manifest changed after capture",
        )
    except (OSError, TypeError):
        checks.require(False, "AOT manifest cannot be re-read")
    toolchain = environment.get("toolchain", {})
    checks.require(
        all(
            isinstance(toolchain.get(tool), str) and toolchain[tool]
            for tool in (
                "nvcc",
                "stwo_rustc",
                "stwo_cargo",
                "stwo_cairo_rustc",
                "stwo_cairo_cargo",
            )
        ),
        "toolchain identity incomplete",
    )
    binaries = environment.get("test_binaries", {})
    checks.require(set(binaries) == set(TARGETS), "test binary set")
    for label, target in TARGETS.items():
        record = binaries.get(label, {})
        checks.require(record.get("target") == target, f"{label}: binary target")
        checks.require(HEX64.fullmatch(str(record.get("sha256", ""))) is not None, f"{label}: binary SHA-256")
        try:
            checks.require(
                sha256(pathlib.Path(record.get("path", ""))) == record.get("sha256"),
                f"{label}: test binary changed after capture",
            )
        except (OSError, TypeError):
            checks.require(False, f"{label}: test binary cannot be re-read")
    relation_value = checks.json(run / "relation.receipt.json", "relation")
    quotient_value = checks.json(run / "quotient.receipt.json", "quotient")
    composition_value = checks.json(run / "composition.receipt.json", "composition")
    metrics = {
        "relation_speedups": relation(checks, relation_value, str(source.get("stwo_head", ""))),
        "quotient_speedups": quotient(checks, quotient_value, str(source.get("stwo_head", ""))),
        "composition_diagnostic_candidate_over_wave": composition(checks, composition_value),
    }
    same_source_positive_speedup = {
        "relation": set(metrics["relation_speedups"]) == {"eager", "captured"}
        and all(value > 1.0 for value in metrics["relation_speedups"].values()),
        "quotient": len(metrics["quotient_speedups"]) == 4
        and all(value > 1.0 for value in metrics["quotient_speedups"].values()),
        "composition": None,
    }
    return {
        "schema": "stwo.sn2-5mhz.cheap-gpu-ab.v1",
        "passed": not checks.errors,
        "benchmark_class": "A40 sm_86 same-source differential proxy",
        "target_useful_mhz": 5.0,
        "sn2_wall_ceiling_seconds": 1.5413728,
        "promotion_credit": {
            "relation": False,
            "quotient": False,
            "composition": False,
            "end_to_end_sn2": False,
        },
        "candidate_admission": {
            "relation_same_source_screen": same_source_positive_speedup["relation"],
            "quotient_same_source_screen": same_source_positive_speedup["quotient"],
            "composition_diagnostic_screen": len(
                metrics["composition_diagnostic_candidate_over_wave"]
            )
            == 2,
        },
        "same_source_positive_speedup": same_source_positive_speedup,
        "source": source,
        "aot_manifest_sha256": environment.get("aot_manifest", {}).get("sha256"),
        "test_binary_sha256": {
            label: binaries.get(label, {}).get("sha256") for label in TARGETS
        },
        "metrics": metrics,
        "errors": checks.errors,
    }


def validate_command(args: argparse.Namespace) -> int:
    record = validate_run(args.run)
    atomic_json(args.out, record)
    print(json.dumps(record, sort_keys=True))
    return int(not record["passed"])


def self_test() -> int:
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        executable = root / "example-test"
        executable.write_bytes(b"test")
        executable.chmod(0o755)
        cargo = root / "cargo.json"
        cargo.write_text(
            json.dumps(
                {
                    "reason": "compiler-artifact",
                    "target": {"name": "example", "kind": ["test"]},
                    "executable": str(executable),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        assert cargo_artifact(cargo, "example") == executable.resolve()
        assert validate_run(root)["passed"] is False
        destination = root / "atomic.json"
        atomic_json(destination, {"passed": True})
        try:
            atomic_json(destination, {"passed": False})
        except FileExistsError:
            pass
        else:
            raise AssertionError("atomic publication replaced existing evidence")
    print("sn2 5mhz cheap-GPU validator self-test: PASS")
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    subcommands = root.add_subparsers(dest="command", required=True)
    artifact = subcommands.add_parser("artifact")
    artifact.add_argument("--cargo-json", type=pathlib.Path, required=True)
    artifact.add_argument("--target", required=True)
    artifact.add_argument("--out", type=pathlib.Path, required=True)
    capture = subcommands.add_parser("capture")
    capture.add_argument("--run", type=pathlib.Path, required=True)
    capture.add_argument("--stwo", type=pathlib.Path, required=True)
    capture.add_argument("--cairo", type=pathlib.Path, required=True)
    capture.add_argument("--aot-manifest", type=pathlib.Path, required=True)
    capture.add_argument("--out", type=pathlib.Path, required=True)
    validate = subcommands.add_parser("validate")
    validate.add_argument("--run", type=pathlib.Path, required=True)
    validate.add_argument("--out", type=pathlib.Path, required=True)
    subcommands.add_parser("self-test")
    return root


def main() -> int:
    args = parser().parse_args()
    if args.command == "artifact":
        return artifact_command(args)
    if args.command == "capture":
        return capture_command(args)
    if args.command == "validate":
        return validate_command(args)
    return self_test()


if __name__ == "__main__":
    sys.exit(main())
