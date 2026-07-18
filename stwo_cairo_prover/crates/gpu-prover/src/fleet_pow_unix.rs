//! Fixed-frame Unix transport for the same-node two-rank PoW path.

use std::io::{self, Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::time::Duration;

use crate::fleet_pow_runtime::{
    FleetPowRankRequest, FleetPowRankResponse, FleetPowRuntimeError, FleetPowTransport,
    FLEET_POW_REQUEST_BYTES, FLEET_POW_RESPONSE_BYTES,
};
use crate::fleet_pow_worker::{FleetPowWorker, FleetPowWorkerError};

const IO_TIMEOUT: Duration = Duration::from_secs(120);

pub struct FleetPowUnixTransport {
    stream: UnixStream,
}

impl FleetPowUnixTransport {
    pub fn connect(path: impl AsRef<Path>) -> io::Result<Self> {
        let stream = UnixStream::connect(path)?;
        configure_timeouts(&stream)?;
        Ok(Self::from_stream(stream))
    }

    pub fn from_stream(stream: UnixStream) -> Self {
        Self { stream }
    }
}

impl FleetPowTransport for FleetPowUnixTransport {
    fn send(&mut self, request: &FleetPowRankRequest) -> Result<(), FleetPowRuntimeError> {
        self.stream
            .write_all(&request.to_bytes())
            .map_err(transport_error)
    }

    fn receive(&mut self) -> Result<FleetPowRankResponse, FleetPowRuntimeError> {
        let mut frame = [0; FLEET_POW_RESPONSE_BYTES];
        self.stream
            .read_exact(&mut frame)
            .map_err(transport_error)?;
        FleetPowRankResponse::from_bytes(&frame)
    }
}

/// Accept one coordinator and serve requests until its clean EOF.
pub fn serve_pow_worker(
    listener: UnixListener,
    worker: &mut FleetPowWorker,
) -> Result<(), FleetPowUnixError> {
    let (mut stream, _) = listener.accept()?;
    configure_timeouts(&stream)?;
    while let Some(frame) = read_frame::<FLEET_POW_REQUEST_BYTES>(&mut stream)? {
        let request = FleetPowRankRequest::from_bytes(&frame)?;
        let response = worker.execute(&request)?;
        stream.write_all(&response.to_bytes())?;
    }
    Ok(())
}

fn configure_timeouts(stream: &UnixStream) -> io::Result<()> {
    stream.set_read_timeout(Some(IO_TIMEOUT))?;
    stream.set_write_timeout(Some(IO_TIMEOUT))
}

fn read_frame<const N: usize>(
    reader: &mut impl Read,
) -> Result<Option<[u8; N]>, FleetPowUnixError> {
    let mut frame = [0; N];
    let mut read = 0;
    while read != N {
        match reader.read(&mut frame[read..])? {
            0 if read == 0 => return Ok(None),
            0 => {
                return Err(FleetPowUnixError::PartialFrame {
                    expected: N,
                    actual: read,
                })
            }
            count => read += count,
        }
    }
    Ok(Some(frame))
}

fn transport_error(error: io::Error) -> FleetPowRuntimeError {
    FleetPowRuntimeError::Transport(error.to_string())
}

#[derive(Debug)]
pub enum FleetPowUnixError {
    Io(io::Error),
    PartialFrame { expected: usize, actual: usize },
    Runtime(FleetPowRuntimeError),
    Worker(FleetPowWorkerError),
}

impl core::fmt::Display for FleetPowUnixError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "fleet PoW Unix worker failed: {self:?}")
    }
}

impl std::error::Error for FleetPowUnixError {}

impl From<io::Error> for FleetPowUnixError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<FleetPowRuntimeError> for FleetPowUnixError {
    fn from(value: FleetPowRuntimeError) -> Self {
        Self::Runtime(value)
    }
}

impl From<FleetPowWorkerError> for FleetPowUnixError {
    fn from(value: FleetPowWorkerError) -> Self {
        Self::Worker(value)
    }
}

#[cfg(test)]
mod tests {
    use std::thread;

    use super::*;

    #[test]
    fn coordinator_round_trips_exact_fixed_frames() {
        let request = request();
        let response = FleetPowRankResponse::completed(&request, Some(256));
        let (coordinator, mut worker) = UnixStream::pair().unwrap();
        let expected_request = request.to_bytes();
        let response_bytes = response.to_bytes();
        let peer = thread::spawn(move || {
            let mut received = [0; FLEET_POW_REQUEST_BYTES];
            worker.read_exact(&mut received).unwrap();
            assert_eq!(received, expected_request);
            worker.write_all(&response_bytes).unwrap();
        });

        let mut transport = FleetPowUnixTransport::from_stream(coordinator);
        transport.send(&request).unwrap();
        assert_eq!(transport.receive().unwrap(), response);
        peer.join().unwrap();
    }

    #[test]
    fn clean_eof_exits_and_partial_frames_fail_closed() {
        let (writer, mut reader) = UnixStream::pair().unwrap();
        drop(writer);
        assert!(read_frame::<FLEET_POW_REQUEST_BYTES>(&mut reader)
            .unwrap()
            .is_none());

        let (mut writer, mut reader) = UnixStream::pair().unwrap();
        writer.write_all(&[7; 13]).unwrap();
        drop(writer);
        assert!(matches!(
            read_frame::<FLEET_POW_REQUEST_BYTES>(&mut reader),
            Err(FleetPowUnixError::PartialFrame {
                expected: FLEET_POW_REQUEST_BYTES,
                actual: 13
            })
        ));

        let (coordinator, mut worker) = UnixStream::pair().unwrap();
        worker.write_all(&[9; 13]).unwrap();
        drop(worker);
        assert!(matches!(
            FleetPowUnixTransport::from_stream(coordinator).receive(),
            Err(FleetPowRuntimeError::Transport(_))
        ));
    }

    fn request() -> FleetPowRankRequest {
        let mut frame = [0; FLEET_POW_REQUEST_BYTES];
        frame[..8].copy_from_slice(b"STW2POW1");
        frame[8..10].copy_from_slice(&1u16.to_le_bytes());
        frame[10] = 1;
        frame[11] = 1;
        frame[12..14].copy_from_slice(&1u16.to_le_bytes());
        frame[14..16].copy_from_slice(&2u16.to_le_bytes());
        frame[20..24].copy_from_slice(&256u32.to_le_bytes());
        frame[24..28].copy_from_slice(&1u32.to_le_bytes());
        frame[32..40].copy_from_slice(&1u64.to_le_bytes());
        frame[56..64].copy_from_slice(&1024u64.to_le_bytes());
        frame[64..96].fill(7);
        FleetPowRankRequest::from_bytes(&frame).unwrap()
    }
}
