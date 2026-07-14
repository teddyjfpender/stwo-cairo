#![forbid(unsafe_code)]

use std::ffi::{OsStr, OsString};
use std::fs::OpenOptions;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::rc::Rc;

use cairo_program_runner_lib::tasks::create_pie_task;
use cairo_program_runner_lib::types::{HashFunc, RunMode};
use cairo_program_runner_lib::{cairo_run_program, ProgramInput, SimpleBootloaderInput, TaskSpec};
use cairo_vm::types::layout_name::LayoutName;
use cairo_vm::types::program::Program;
use serde::Serialize;
use stwo_cairo_adapter::adapter::adapt;

const USAGE: &str = "expected exactly: --pie PATH --backend simd --engine legacy --adapt-only";
const OUTPUT_BUFFER_BYTES: usize = 1024 * 1024;

#[derive(Debug, PartialEq, Eq)]
struct Protocol {
    pie: PathBuf,
    bootloader: PathBuf,
    output: PathBuf,
}

impl Protocol {
    fn from_process() -> Result<Self, String> {
        let args: Vec<_> = std::env::args_os().skip(1).collect();
        let pie = parse_argv(&args)?;
        Ok(Self {
            pie,
            bootloader: required_path("STWO_BOOTLOADER_JSON")?,
            output: required_path("STWO_DUMP_INPUT")?,
        })
    }
}

fn parse_argv(args: &[OsString]) -> Result<PathBuf, String> {
    let exact = args.len() == 7
        && args[0] == OsStr::new("--pie")
        && args[2] == OsStr::new("--backend")
        && args[3] == OsStr::new("simd")
        && args[4] == OsStr::new("--engine")
        && args[5] == OsStr::new("legacy")
        && args[6] == OsStr::new("--adapt-only")
        && !args[1].is_empty();
    exact
        .then(|| PathBuf::from(&args[1]))
        .ok_or_else(|| USAGE.into())
}

fn required_path(name: &str) -> Result<PathBuf, String> {
    std::env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .ok_or_else(|| format!("{name} must name a path"))
}

fn adapt_pie(protocol: &Protocol) -> Result<stwo_cairo_adapter::ProverInput, String> {
    let task = create_pie_task(&protocol.pie)
        .map(Rc::new)
        .map_err(|error| format!("failed to load PIE {}: {error:?}", protocol.pie.display()))?;
    let input = SimpleBootloaderInput {
        fact_topologies_path: None,
        single_page: true,
        tasks: vec![TaskSpec {
            task,
            program_hash_function: HashFunc::Blake,
        }],
    };
    let bootloader = Program::from_file(&protocol.bootloader, Some("main")).map_err(|error| {
        format!(
            "failed to load bootloader {}: {error}",
            protocol.bootloader.display()
        )
    })?;
    let config = RunMode::Proof {
        layout: LayoutName::all_cairo_stwo,
        dynamic_layout_params: None,
        disable_trace_padding: true,
        relocate_mem: false,
    }
    .create_config();
    let runner = cairo_run_program(
        &bootloader,
        Some(ProgramInput::Value(Box::new(input))),
        config,
        None,
    )
    .map_err(|error| format!("failed to run PIE through bootloader: {error}"))?;
    adapt(&runner).map_err(|error| format!("failed to adapt runner: {error}"))
}

struct CountingWriter<W> {
    inner: W,
    bytes: u64,
}

impl<W> CountingWriter<W> {
    const fn new(inner: W) -> Self {
        Self { inner, bytes: 0 }
    }
}

impl<W: Write> Write for CountingWriter<W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let written = self.inner.write(bytes)?;
        self.bytes = self
            .bytes
            .checked_add(written as u64)
            .ok_or_else(|| io::Error::other("serialized byte count overflow"))?;
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

fn serialize_bincode_v1<T: Serialize, W: Write>(value: &T, writer: W) -> Result<u64, String> {
    let mut writer = CountingWriter::new(writer);
    // The bincode 1.x top-level helper is the legacy little-endian fixed-int format
    // used by gpu_bench's original STWO_DUMP_INPUT path.
    bincode::serialize_into(&mut writer, value)
        .map_err(|error| format!("failed to serialize ProverInput: {error}"))?;
    writer
        .flush()
        .map_err(|error| format!("failed to flush ProverInput: {error}"))?;
    Ok(writer.bytes)
}

fn write_input<T: Serialize>(path: &Path, value: &T) -> Result<u64, String> {
    let file = OpenOptions::new()
        .write(true)
        .truncate(true)
        .open(path)
        .map_err(|error| format!("failed to open output {}: {error}", path.display()))?;
    let mut writer = BufWriter::with_capacity(OUTPUT_BUFFER_BYTES, file);
    let bytes = serialize_bincode_v1(value, &mut writer)?;
    writer
        .get_ref()
        .sync_all()
        .map_err(|error| format!("failed to sync output {}: {error}", path.display()))?;
    Ok(bytes)
}

fn run() -> Result<(), String> {
    let protocol = Protocol::from_process()?;
    let input = adapt_pie(&protocol)?;
    let bytes = write_input(&protocol.output, &input)?;
    eprintln!(
        "prover input dumped: {bytes} bytes -> {}",
        protocol.output.display()
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Serialize)]
    enum Variant {
        Value(u32),
    }

    #[derive(Serialize)]
    struct Representative {
        count: usize,
        signed: i64,
        words: Vec<u32>,
        variant: Variant,
    }

    #[derive(Default)]
    struct FragmentedWriter(Vec<u8>);

    impl Write for FragmentedWriter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            let count = bytes.len().min(3);
            self.0.extend_from_slice(&bytes[..count]);
            Ok(count)
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    fn os_args(values: &[&str]) -> Vec<OsString> {
        values.iter().map(OsString::from).collect()
    }

    #[test]
    fn argv_accepts_only_the_sealed_protocol() {
        let exact = os_args(&[
            "--pie",
            "/proc/self/fd/4",
            "--backend",
            "simd",
            "--engine",
            "legacy",
            "--adapt-only",
        ]);
        assert_eq!(parse_argv(&exact).unwrap(), Path::new("/proc/self/fd/4"));

        for invalid in [
            os_args(&[
                "--pie",
                "/tmp/input.zip",
                "--backend",
                "cuda",
                "--engine",
                "legacy",
                "--adapt-only",
            ]),
            os_args(&[
                "--backend",
                "simd",
                "--pie",
                "/tmp/input.zip",
                "--engine",
                "legacy",
                "--adapt-only",
            ]),
            os_args(&[
                "--pie",
                "",
                "--backend",
                "simd",
                "--engine",
                "legacy",
                "--adapt-only",
            ]),
            os_args(&[
                "--pie",
                "/tmp/input.zip",
                "--backend",
                "simd",
                "--engine",
                "legacy",
                "--adapt-only",
                "--extra",
            ]),
        ] {
            assert_eq!(parse_argv(&invalid).unwrap_err(), USAGE);
        }
    }

    #[test]
    fn streaming_bytes_equal_gpu_bench_bincode_v1_bytes() {
        let value = Representative {
            count: usize::MAX / 3,
            signed: -0x0102_0304_0506_0708,
            words: vec![0, 1, u32::MAX],
            variant: Variant::Value(0x1122_3344),
        };
        let expected = bincode::serialize(&value).unwrap();
        let mut output = FragmentedWriter::default();
        let count = serialize_bincode_v1(&value, &mut output).unwrap();
        assert_eq!(output.0, expected);
        assert_eq!(count, expected.len() as u64);
    }
}
