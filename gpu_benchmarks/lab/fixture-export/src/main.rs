mod artifact_io;
mod fri_round6;
mod fri_round6_capture;
mod fri_round6_index;
mod fri_round6_io;
mod fri_round6_provenance;
#[cfg(test)]
mod fri_round6_provenance_tests;
mod fri_round6_transcript;
mod fri_round6_validation;
mod legacy;
mod model;
mod oracle;
mod pedersen_builtin_semantics;
mod producer;
mod semantic_support;
mod source_snapshot;
mod streaming;
#[cfg(test)]
mod streaming_tests;

use std::env;
use std::path::{Path, PathBuf};
use std::process;

use serde::Deserialize;

#[derive(Debug)]
struct Args {
    fixture: Option<PathBuf>,
    prover_input: Option<PathBuf>,
    prover_input_sha256: Option<String>,
    expected_exporter_sha256: Option<String>,
    fixture_class: Option<String>,
    fixture_index: Option<PathBuf>,
    fri_round6_capture: Option<PathBuf>,
    fri_round6_capture_sha256: Option<String>,
    fri_round6_output_dir: Option<PathBuf>,
    fri_round6_synthetic_output_dir: Option<PathBuf>,
    fri_round6_validate_dir: Option<PathBuf>,
    fri_round6_provenance: Option<PathBuf>,
    fri_round6_provenance_sha256: Option<String>,
    output: Option<PathBuf>,
    check: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut fixture = None;
    let mut prover_input = None;
    let mut prover_input_sha256 = None;
    let mut expected_exporter_sha256 = None;
    let mut fixture_class = None;
    let mut fixture_index = None;
    let mut fri_round6_capture = None;
    let mut fri_round6_capture_sha256 = None;
    let mut fri_round6_output_dir = None;
    let mut fri_round6_synthetic_output_dir = None;
    let mut fri_round6_validate_dir = None;
    let mut fri_round6_provenance = None;
    let mut fri_round6_provenance_sha256 = None;
    let mut output = None;
    let mut check = false;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--fixture" => set_path(&mut fixture, args.next(), "--fixture")?,
            "--prover-input" => set_path(&mut prover_input, args.next(), "--prover-input")?,
            "--prover-input-sha256" => set_string(
                &mut prover_input_sha256,
                args.next(),
                "--prover-input-sha256",
            )?,
            "--expected-exporter-sha256" => set_string(
                &mut expected_exporter_sha256,
                args.next(),
                "--expected-exporter-sha256",
            )?,
            "--fixture-class" => set_string(&mut fixture_class, args.next(), "--fixture-class")?,
            "--fixture-index" => set_path(&mut fixture_index, args.next(), "--fixture-index")?,
            "--fri-round6-captured-unsealed-output-dir" => set_path(
                &mut fri_round6_output_dir,
                args.next(),
                "--fri-round6-captured-unsealed-output-dir",
            )?,
            "--fri-round6-capture" => {
                set_path(&mut fri_round6_capture, args.next(), "--fri-round6-capture")?
            }
            "--fri-round6-capture-sha256" => set_string(
                &mut fri_round6_capture_sha256,
                args.next(),
                "--fri-round6-capture-sha256",
            )?,
            "--fri-round6-synthetic-layout-output-dir" => set_path(
                &mut fri_round6_synthetic_output_dir,
                args.next(),
                "--fri-round6-synthetic-layout-output-dir",
            )?,
            "--validate-fri-round6-captured-unsealed-dir" => set_path(
                &mut fri_round6_validate_dir,
                args.next(),
                "--validate-fri-round6-captured-unsealed-dir",
            )?,
            "--preflight-fri-round6-provenance" => set_path(
                &mut fri_round6_provenance,
                args.next(),
                "--preflight-fri-round6-provenance",
            )?,
            "--fri-round6-provenance-sha256" => set_string(
                &mut fri_round6_provenance_sha256,
                args.next(),
                "--fri-round6-provenance-sha256",
            )?,
            "--output" => set_path(&mut output, args.next(), "--output")?,
            "--check" => check = true,
            "--help" | "-h" => {
                println!(
                    "usage:\n  stwo-gpu-lab-fixture-export --fixture PATH [--check] [--output PATH]\n  stwo-gpu-lab-fixture-export --prover-input PATH --prover-input-sha256 SHA256 --expected-exporter-sha256 SHA256 --fixture-class CLASS --fixture-index PATH --output PATH\n  stwo-gpu-lab-fixture-export --fri-round6-capture PATH --fri-round6-capture-sha256 SHA256 --expected-exporter-sha256 SHA256 --fri-round6-captured-unsealed-output-dir PATH\n  stwo-gpu-lab-fixture-export --validate-fri-round6-captured-unsealed-dir PATH --fri-round6-capture PATH --fri-round6-capture-sha256 SHA256 --expected-exporter-sha256 SHA256\n  stwo-gpu-lab-fixture-export --preflight-fri-round6-provenance PATH --fri-round6-provenance-sha256 SHA256\n  stwo-gpu-lab-fixture-export --fri-round6-synthetic-layout-output-dir PATH"
                );
                process::exit(0);
            }
            _ => return Err(format!("unknown argument: {arg}")),
        }
    }
    Ok(Args {
        fixture,
        prover_input,
        prover_input_sha256,
        expected_exporter_sha256,
        fixture_class,
        fixture_index,
        fri_round6_capture,
        fri_round6_capture_sha256,
        fri_round6_output_dir,
        fri_round6_synthetic_output_dir,
        fri_round6_validate_dir,
        fri_round6_provenance,
        fri_round6_provenance_sha256,
        output,
        check,
    })
}

fn set_string(slot: &mut Option<String>, value: Option<String>, flag: &str) -> Result<(), String> {
    if slot.is_some() {
        return Err(format!("duplicate {flag}"));
    }
    let value = value.ok_or_else(|| format!("{flag} requires a value"))?;
    if value.is_empty() {
        return Err(format!("{flag} cannot be empty"));
    }
    *slot = Some(value);
    Ok(())
}

fn set_path(slot: &mut Option<PathBuf>, value: Option<String>, flag: &str) -> Result<(), String> {
    if slot.is_some() {
        return Err(format!("duplicate {flag}"));
    }
    *slot = Some(PathBuf::from(
        value.ok_or_else(|| format!("{flag} requires a path"))?,
    ));
    Ok(())
}

fn schema_version(path: &Path) -> Result<String, String> {
    #[derive(Deserialize)]
    struct Header {
        schema_version: String,
    }
    let (bytes, _) = model::load_bounded(path, streaming::MAX_INDEX_BYTES, "fixture JSON/index")?;
    serde_json::from_slice::<Header>(&bytes)
        .map(|header| header.schema_version)
        .map_err(|error| format!("parse fixture header {}: {error}", path.display()))
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    if args.fri_round6_provenance.is_some() || args.fri_round6_provenance_sha256.is_some() {
        if args.fixture.is_some()
            || args.prover_input.is_some()
            || args.prover_input_sha256.is_some()
            || args.expected_exporter_sha256.is_some()
            || args.fixture_class.is_some()
            || args.fixture_index.is_some()
            || args.fri_round6_capture.is_some()
            || args.fri_round6_capture_sha256.is_some()
            || args.fri_round6_output_dir.is_some()
            || args.fri_round6_synthetic_output_dir.is_some()
            || args.fri_round6_validate_dir.is_some()
            || args.output.is_some()
            || args.check
        {
            return Err("FRI provenance preflight is a standalone non-admission mode".into());
        }
        let verified = fri_round6_provenance::preflight(
            args.fri_round6_provenance
                .as_deref()
                .ok_or("provenance preflight requires --preflight-fri-round6-provenance")?,
            args.fri_round6_provenance_sha256
                .as_deref()
                .ok_or("provenance preflight requires --fri-round6-provenance-sha256")?,
        )?;
        println!(
            "{} manifest_sha256={} proof_shape_sha256={}",
            fri_round6_provenance::IDENTITY_PREFLIGHT_STATUS,
            verified.manifest_sha256,
            verified.proof_shape_sha256
        );
        return Ok(());
    }
    if let Some(output_dir) = args.fri_round6_synthetic_output_dir.as_deref() {
        if args.fixture.is_some()
            || args.prover_input.is_some()
            || args.prover_input_sha256.is_some()
            || args.expected_exporter_sha256.is_some()
            || args.fixture_class.is_some()
            || args.fixture_index.is_some()
            || args.fri_round6_capture.is_some()
            || args.fri_round6_capture_sha256.is_some()
            || args.fri_round6_output_dir.is_some()
            || args.fri_round6_validate_dir.is_some()
            || args.output.is_some()
            || args.check
        {
            return Err(
                "--fri-round6-synthetic-layout-output-dir is a standalone self-test mode".into(),
            );
        }
        return fri_round6_io::export_synthetic(output_dir);
    }
    if let Some(output_dir) = args.fri_round6_validate_dir.as_deref() {
        if args.fixture.is_some()
            || args.prover_input.is_some()
            || args.prover_input_sha256.is_some()
            || args.fixture_class.is_some()
            || args.fixture_index.is_some()
            || args.fri_round6_output_dir.is_some()
            || args.fri_round6_synthetic_output_dir.is_some()
            || args.output.is_some()
            || args.check
        {
            return Err(
                "--validate-fri-round6-captured-unsealed-dir is a standalone validation mode"
                    .into(),
            );
        }
        let capture = fri_round6_capture::load(
            args.fri_round6_capture
                .as_deref()
                .ok_or("FRI validation requires --fri-round6-capture")?,
            args.fri_round6_capture_sha256
                .as_deref()
                .ok_or("FRI validation requires --fri-round6-capture-sha256")?,
        )?;
        return fri_round6_io::validate_capture_dir(
            output_dir,
            &capture,
            args.expected_exporter_sha256
                .as_deref()
                .ok_or("FRI validation requires --expected-exporter-sha256")?,
        );
    }
    if args.fri_round6_capture.is_some()
        || args.fri_round6_capture_sha256.is_some()
        || args.fri_round6_output_dir.is_some()
    {
        if args.fixture.is_some()
            || args.prover_input.is_some()
            || args.prover_input_sha256.is_some()
            || args.fixture_class.is_some()
            || args.fixture_index.is_some()
            || args.output.is_some()
            || args.check
        {
            return Err("FRI round-6 capture export is a standalone mode".into());
        }
        let capture = fri_round6_capture::load(
            args.fri_round6_capture
                .as_deref()
                .ok_or("capture export requires --fri-round6-capture")?,
            args.fri_round6_capture_sha256
                .as_deref()
                .ok_or("capture export requires --fri-round6-capture-sha256")?,
        )?;
        return fri_round6_io::export_capture(
            args.fri_round6_output_dir
                .as_deref()
                .ok_or("capture export requires --fri-round6-captured-unsealed-output-dir")?,
            &capture,
            args.expected_exporter_sha256
                .as_deref()
                .ok_or("capture export requires --expected-exporter-sha256")?,
        );
    }
    match (args.fixture.as_deref(), args.prover_input.as_deref()) {
        (Some(fixture), None) => {
            if args.prover_input_sha256.is_some()
                || args.expected_exporter_sha256.is_some()
                || args.fixture_class.is_some()
                || args.fixture_index.is_some()
            {
                return Err(
                    "--prover-input-sha256/--fixture-class/--fixture-index require --prover-input"
                        .into(),
                );
            }
            match schema_version(fixture)?.as_str() {
                model::INLINE_SCHEMA => legacy::export(fixture, args.output.as_deref(), args.check),
                model::INDEX_SCHEMA => Err(
                    "production fixture indexes can only be generated and checked from a pinned --prover-input"
                        .into(),
                ),
                schema => Err(format!("unsupported fixture schema: {schema}")),
            }
        }
        (None, Some(prover_input)) => {
            if args.check {
                return Err("real fixture generation is always fail-closed; omit --check".into());
            }
            let expected_exporter_sha256 = args
                .expected_exporter_sha256
                .as_deref()
                .ok_or("--prover-input requires --expected-exporter-sha256")?;
            model::validate_sha256(
                expected_exporter_sha256,
                "expected exporter executable sha256",
            )?;
            let exporter_executable_sha256 = model::current_executable_sha256()?;
            if exporter_executable_sha256 != expected_exporter_sha256 {
                return Err(format!(
                    "exporter executable sha256 {exporter_executable_sha256} != required {expected_exporter_sha256}; refusing before fixture writes"
                ));
            }
            producer::generate(
                prover_input,
                args.prover_input_sha256
                    .as_deref()
                    .ok_or("--prover-input requires --prover-input-sha256")?,
                args.fixture_class
                    .as_deref()
                    .ok_or("--prover-input requires --fixture-class")?,
                args.fixture_index
                    .as_deref()
                    .ok_or("--prover-input requires --fixture-index")?,
                args.output
                    .as_deref()
                    .ok_or("--prover-input requires --output")?,
                &exporter_executable_sha256,
            )?;
            if model::current_executable_sha256()? != exporter_executable_sha256 {
                return Err("exporter executable changed during fixture generation".into());
            }
            Ok(())
        }
        (Some(_), Some(_)) => Err("choose exactly one of --fixture or --prover-input".into()),
        (None, None) => {
            Err("missing --fixture, --prover-input, or an explicit FRI round-6 output mode".into())
        }
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("fixture-export: {error}");
        process::exit(1);
    }
}
