use std::os::unix::net::UnixListener;
use std::process::ExitCode;

use stwo_cairo_gpu_prover::fleet_plan::WorkerId;
use stwo_cairo_gpu_prover::fleet_pow_unix::serve_pow_worker;
use stwo_cairo_gpu_prover::fleet_pow_worker::FleetPowWorker;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("fleet_pow_worker: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let socket = args
        .next()
        .ok_or_else(|| "usage: fleet_pow_worker <socket> <pow-bits>".to_owned())?;
    let pow_bits = args
        .next()
        .ok_or_else(|| "usage: fleet_pow_worker <socket> <pow-bits>".to_owned())?
        .parse::<u32>()
        .map_err(|error| format!("invalid pow bits: {error}"))?;
    if args.next().is_some() {
        return Err("usage: fleet_pow_worker <socket> <pow-bits>".to_owned());
    }

    let mut worker =
        FleetPowWorker::new(WorkerId(1), pow_bits).map_err(|error| error.to_string())?;
    let listener = UnixListener::bind(&socket).map_err(|error| error.to_string())?;
    let result = serve_pow_worker(listener, &mut worker).map_err(|error| error.to_string());
    let _ = std::fs::remove_file(socket);
    result
}
