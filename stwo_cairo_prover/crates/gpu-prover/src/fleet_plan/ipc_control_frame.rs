//! Authenticated byte framing for one rank's CUDA-IPC install channel.
//!
//! Frames are address-free control data suitable for a Unix socketpair. CUDA
//! handles remain opaque payload bytes owned and validated by the existing IPC
//! descriptor types; this module never opens a context or handle.

use stwo_backend_cuda::IpcExchangeInstallDomain;

use super::{FleetRuntimeView, WorkerId};

const MAGIC: &[u8; 8] = b"STWOIPCF";
const PROTOCOL_VERSION: u16 = 1;
const HEADER_BYTES: usize = 136;
const MAC_BYTES: usize = 32;
const MAC_DOMAIN: &[u8] = b"stwo-cairo.fleet-ipc-control-frame.v1\0";
const KEY_DERIVATION_DOMAIN: &[u8] = b"stwo-cairo.fleet-ipc-control-key.v1\0";

const VERSION_OFFSET: usize = 8;
const KIND_OFFSET: usize = 10;
const RANK_OFFSET: usize = 12;
const HEADER_BYTES_OFFSET: usize = 14;
const GENERATION_OFFSET: usize = 16;
const SEQUENCE_OFFSET: usize = 24;
const PAYLOAD_BYTES_OFFSET: usize = 32;
const PLAN_IDENTITY_OFFSET: usize = 40;
const INSTALL_DOMAIN_OFFSET: usize = 72;
const INSTALL_NONCE_OFFSET: usize = 104;

/// Exact install-channel payload type. A known kind is authenticated but its
/// payload remains under the existing descriptor/acknowledgement authority.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u16)]
pub(crate) enum FleetIpcControlMessageKind {
    RankDescriptorStatement = 1,
    DescriptorBundle = 2,
    InstallAcknowledgement = 3,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FleetIpcControlEndpointRole {
    Controller,
    Rank,
}

impl FleetIpcControlEndpointRole {
    const fn send_direction(self) -> FleetIpcControlDirection {
        match self {
            Self::Controller => FleetIpcControlDirection::ControllerToRank,
            Self::Rank => FleetIpcControlDirection::RankToController,
        }
    }

    const fn receive_direction(self) -> FleetIpcControlDirection {
        match self {
            Self::Controller => FleetIpcControlDirection::RankToController,
            Self::Rank => FleetIpcControlDirection::ControllerToRank,
        }
    }
}

#[derive(Clone, Copy)]
enum FleetIpcControlDirection {
    ControllerToRank,
    RankToController,
}

impl FleetIpcControlDirection {
    const fn domain(self) -> &'static [u8] {
        match self {
            Self::ControllerToRank => b"controller-to-rank",
            Self::RankToController => b"rank-to-controller",
        }
    }

    const fn allows(self, kind: FleetIpcControlMessageKind) -> bool {
        match self {
            Self::ControllerToRank => {
                matches!(kind, FleetIpcControlMessageKind::DescriptorBundle)
            }
            Self::RankToController => matches!(
                kind,
                FleetIpcControlMessageKind::RankDescriptorStatement
                    | FleetIpcControlMessageKind::InstallAcknowledgement
            ),
        }
    }
}

impl FleetIpcControlMessageKind {
    fn from_wire(value: u16) -> Result<Self, FleetIpcControlFrameError> {
        match value {
            1 => Ok(Self::RankDescriptorStatement),
            2 => Ok(Self::DescriptorBundle),
            3 => Ok(Self::InstallAcknowledgement),
            _ => Err(FleetIpcControlFrameError::UnknownMessageKind(value)),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FleetIpcControlFrameError {
    ZeroInstallNonce,
    ZeroMacKey,
    UnknownRank(WorkerId),
    ProofGenerationOverflow,
    MessageKindForWrongDirection(FleetIpcControlMessageKind),
    PayloadLengthOverflow,
    SequenceExhausted,
    Truncated {
        expected: usize,
        actual: usize,
    },
    TrailingBytes {
        expected: usize,
        actual: usize,
    },
    InvalidMagic,
    UnknownProtocolVersion(u16),
    UnknownMessageKind(u16),
    InvalidHeaderLength(u16),
    AuthenticationFailed,
    PlanIdentityMismatch,
    ProofGenerationMismatch,
    InstallDomainMismatch,
    InstallNonceMismatch,
    RankMismatch {
        expected: WorkerId,
        actual: WorkerId,
    },
    SequenceMismatch {
        expected: u64,
        actual: u64,
    },
}

impl core::fmt::Display for FleetIpcControlFrameError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "invalid fleet IPC control frame: {self:?}")
    }
}

impl std::error::Error for FleetIpcControlFrameError {}

/// One authenticated message borrowed directly from the socket frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct FleetIpcControlMessage<'frame> {
    kind: FleetIpcControlMessageKind,
    sequence: u64,
    payload: &'frame [u8],
}

impl<'frame> FleetIpcControlMessage<'frame> {
    pub(crate) const fn kind(&self) -> FleetIpcControlMessageKind {
        self.kind
    }

    pub(crate) const fn sequence(&self) -> u64 {
        self.sequence
    }

    pub(crate) const fn payload(&self) -> &'frame [u8] {
        self.payload
    }
}

/// Full-duplex sequence and MAC state for one plan/generation/rank socketpair.
///
/// The key must be a fresh secret for the rank channel. It authenticates
/// control bytes only and is never CUDA authority. In-memory key cleanup is
/// best effort; the caller remains responsible for provisioning and erasing
/// any other copies.
pub(crate) struct FleetIpcControlCodec {
    plan_identity: [u8; 32],
    proof_generation: u64,
    install_domain: IpcExchangeInstallDomain,
    install_nonce: [u8; 32],
    rank: WorkerId,
    send_direction: FleetIpcControlDirection,
    receive_direction: FleetIpcControlDirection,
    send_mac_key: SecretKey,
    receive_mac_key: SecretKey,
    next_send_sequence: u64,
    next_receive_sequence: u64,
}

impl FleetIpcControlCodec {
    pub(crate) fn new(
        view: &FleetRuntimeView,
        proof_generation: u64,
        install_domain: IpcExchangeInstallDomain,
        install_nonce: [u8; 32],
        rank: WorkerId,
        role: FleetIpcControlEndpointRole,
        channel_secret: [u8; 32],
    ) -> Result<Self, FleetIpcControlFrameError> {
        let channel_secret = SecretKey(channel_secret);
        if install_nonce == [0; 32] {
            return Err(FleetIpcControlFrameError::ZeroInstallNonce);
        }
        if channel_secret.is_zero() {
            return Err(FleetIpcControlFrameError::ZeroMacKey);
        }
        if proof_generation == u64::MAX {
            return Err(FleetIpcControlFrameError::ProofGenerationOverflow);
        }
        if view
            .exchange_reserves()
            .get(usize::from(rank.0))
            .is_none_or(|reserve| reserve.worker != rank)
        {
            return Err(FleetIpcControlFrameError::UnknownRank(rank));
        }
        let plan_identity = view.plan_identity();
        let send_direction = role.send_direction();
        let receive_direction = role.receive_direction();
        let send_mac_key = derive_direction_key(
            &channel_secret,
            plan_identity,
            proof_generation,
            install_domain,
            install_nonce,
            rank,
            send_direction,
        );
        let receive_mac_key = derive_direction_key(
            &channel_secret,
            plan_identity,
            proof_generation,
            install_domain,
            install_nonce,
            rank,
            receive_direction,
        );
        Ok(Self {
            plan_identity,
            proof_generation,
            install_domain,
            install_nonce,
            rank,
            send_direction,
            receive_direction,
            send_mac_key,
            receive_mac_key,
            next_send_sequence: 0,
            next_receive_sequence: 0,
        })
    }

    /// Seal one exact payload. The returned bytes can be written verbatim to a
    /// Unix stream or sequenced-packet socket.
    pub(crate) fn seal(
        &mut self,
        kind: FleetIpcControlMessageKind,
        payload: &[u8],
    ) -> Result<Vec<u8>, FleetIpcControlFrameError> {
        if !self.send_direction.allows(kind) {
            return Err(FleetIpcControlFrameError::MessageKindForWrongDirection(
                kind,
            ));
        }
        let payload_bytes = u64::try_from(payload.len())
            .map_err(|_| FleetIpcControlFrameError::PayloadLengthOverflow)?;
        let next_sequence = self
            .next_send_sequence
            .checked_add(1)
            .ok_or(FleetIpcControlFrameError::SequenceExhausted)?;
        let frame_bytes = HEADER_BYTES
            .checked_add(payload.len())
            .and_then(|bytes| bytes.checked_add(MAC_BYTES))
            .ok_or(FleetIpcControlFrameError::PayloadLengthOverflow)?;

        let mut frame = Vec::with_capacity(frame_bytes);
        frame.extend_from_slice(MAGIC);
        frame.extend_from_slice(&PROTOCOL_VERSION.to_le_bytes());
        frame.extend_from_slice(&(kind as u16).to_le_bytes());
        frame.extend_from_slice(&self.rank.0.to_le_bytes());
        frame.extend_from_slice(&(HEADER_BYTES as u16).to_le_bytes());
        frame.extend_from_slice(&self.proof_generation.to_le_bytes());
        frame.extend_from_slice(&self.next_send_sequence.to_le_bytes());
        frame.extend_from_slice(&payload_bytes.to_le_bytes());
        frame.extend_from_slice(&self.plan_identity);
        frame.extend_from_slice(self.install_domain.as_bytes());
        frame.extend_from_slice(&self.install_nonce);
        debug_assert_eq!(frame.len(), HEADER_BYTES);
        frame.extend_from_slice(payload);
        let mac = frame_mac(&self.send_mac_key, &frame);
        frame.extend_from_slice(mac.as_bytes());
        debug_assert_eq!(frame.len(), frame_bytes);

        self.next_send_sequence = next_sequence;
        Ok(frame)
    }

    /// Authenticate and admit exactly the next frame. Any rejected frame leaves
    /// receive sequence state unchanged, so a valid retry remains possible.
    pub(crate) fn open<'frame>(
        &mut self,
        frame: &'frame [u8],
    ) -> Result<FleetIpcControlMessage<'frame>, FleetIpcControlFrameError> {
        let minimum = HEADER_BYTES + MAC_BYTES;
        if frame.len() < minimum {
            return Err(FleetIpcControlFrameError::Truncated {
                expected: minimum,
                actual: frame.len(),
            });
        }

        // The MAC is always the final fixed-width field. Authenticate the
        // complete bytes received before trusting any attacker-controlled
        // header value, including its claimed payload/header lengths.
        let received_mac_offset = frame.len() - MAC_BYTES;
        let expected_mac = frame_mac(&self.receive_mac_key, &frame[..received_mac_offset]);
        if !mac_matches(expected_mac, &frame[received_mac_offset..]) {
            return Err(FleetIpcControlFrameError::AuthenticationFailed);
        }

        if &frame[..MAGIC.len()] != MAGIC {
            return Err(FleetIpcControlFrameError::InvalidMagic);
        }
        let version = u16_at(frame, VERSION_OFFSET);
        if version != PROTOCOL_VERSION {
            return Err(FleetIpcControlFrameError::UnknownProtocolVersion(version));
        }
        let kind = FleetIpcControlMessageKind::from_wire(u16_at(frame, KIND_OFFSET))?;
        if !self.receive_direction.allows(kind) {
            return Err(FleetIpcControlFrameError::MessageKindForWrongDirection(
                kind,
            ));
        }
        let header_bytes = u16_at(frame, HEADER_BYTES_OFFSET);
        if usize::from(header_bytes) != HEADER_BYTES {
            return Err(FleetIpcControlFrameError::InvalidHeaderLength(header_bytes));
        }
        let payload_bytes = usize::try_from(u64_at(frame, PAYLOAD_BYTES_OFFSET))
            .map_err(|_| FleetIpcControlFrameError::PayloadLengthOverflow)?;
        let expected_bytes = HEADER_BYTES
            .checked_add(payload_bytes)
            .and_then(|bytes| bytes.checked_add(MAC_BYTES))
            .ok_or(FleetIpcControlFrameError::PayloadLengthOverflow)?;
        if frame.len() < expected_bytes {
            return Err(FleetIpcControlFrameError::Truncated {
                expected: expected_bytes,
                actual: frame.len(),
            });
        }
        if frame.len() > expected_bytes {
            return Err(FleetIpcControlFrameError::TrailingBytes {
                expected: expected_bytes,
                actual: frame.len(),
            });
        }

        let mac_offset = HEADER_BYTES + payload_bytes;
        debug_assert_eq!(mac_offset, received_mac_offset);

        if frame[PLAN_IDENTITY_OFFSET..INSTALL_DOMAIN_OFFSET] != self.plan_identity {
            return Err(FleetIpcControlFrameError::PlanIdentityMismatch);
        }
        if u64_at(frame, GENERATION_OFFSET) != self.proof_generation {
            return Err(FleetIpcControlFrameError::ProofGenerationMismatch);
        }
        if frame[INSTALL_DOMAIN_OFFSET..INSTALL_NONCE_OFFSET] != *self.install_domain.as_bytes() {
            return Err(FleetIpcControlFrameError::InstallDomainMismatch);
        }
        if frame[INSTALL_NONCE_OFFSET..HEADER_BYTES] != self.install_nonce {
            return Err(FleetIpcControlFrameError::InstallNonceMismatch);
        }
        let rank = WorkerId(u16_at(frame, RANK_OFFSET));
        if rank != self.rank {
            return Err(FleetIpcControlFrameError::RankMismatch {
                expected: self.rank,
                actual: rank,
            });
        }
        let sequence = u64_at(frame, SEQUENCE_OFFSET);
        if sequence != self.next_receive_sequence {
            return Err(FleetIpcControlFrameError::SequenceMismatch {
                expected: self.next_receive_sequence,
                actual: sequence,
            });
        }
        let next_sequence = self
            .next_receive_sequence
            .checked_add(1)
            .ok_or(FleetIpcControlFrameError::SequenceExhausted)?;
        self.next_receive_sequence = next_sequence;
        Ok(FleetIpcControlMessage {
            kind,
            sequence,
            payload: &frame[HEADER_BYTES..mac_offset],
        })
    }
}

struct SecretKey([u8; 32]);

impl SecretKey {
    fn is_zero(&self) -> bool {
        self.0 == [0; 32]
    }
}

impl Drop for SecretKey {
    fn drop(&mut self) {
        for byte in &mut self.0 {
            // SAFETY: every pointer comes from an exclusive reference to the
            // live key array. Volatile stores prevent dead-store elimination.
            unsafe { core::ptr::write_volatile(byte, 0) };
        }
        core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    }
}

fn u16_at(frame: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(
        frame[offset..offset + size_of::<u16>()]
            .try_into()
            .expect("fixed frame header was length checked"),
    )
}

fn u64_at(frame: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(
        frame[offset..offset + size_of::<u64>()]
            .try_into()
            .expect("fixed frame header was length checked"),
    )
}

fn derive_direction_key(
    channel_secret: &SecretKey,
    plan_identity: [u8; 32],
    proof_generation: u64,
    install_domain: IpcExchangeInstallDomain,
    install_nonce: [u8; 32],
    rank: WorkerId,
    direction: FleetIpcControlDirection,
) -> SecretKey {
    let mut hasher = blake3::Hasher::new_keyed(&channel_secret.0);
    hasher.update(KEY_DERIVATION_DOMAIN);
    hasher.update(&plan_identity);
    hasher.update(&proof_generation.to_le_bytes());
    hasher.update(install_domain.as_bytes());
    hasher.update(&install_nonce);
    hasher.update(&rank.0.to_le_bytes());
    hasher.update(direction.domain());
    SecretKey(*hasher.finalize().as_bytes())
}

fn frame_mac(key: &SecretKey, frame_without_mac: &[u8]) -> blake3::Hash {
    let mut hasher = blake3::Hasher::new_keyed(&key.0);
    hasher.update(MAC_DOMAIN);
    hasher.update(frame_without_mac);
    hasher.finalize()
}

fn mac_matches(expected: blake3::Hash, actual: &[u8]) -> bool {
    let actual = blake3::Hash::from_bytes(actual.try_into().expect("MAC width checked"));
    // `blake3::Hash::eq` is explicitly constant-time; arrays/slices are not.
    expected == actual
}

#[cfg(test)]
pub(super) fn test_only_remac(frame: &mut [u8], sender: &FleetIpcControlCodec) {
    assert!(frame.len() >= HEADER_BYTES + MAC_BYTES);
    let mac_offset = frame.len() - MAC_BYTES;
    let mac = frame_mac(&sender.send_mac_key, &frame[..mac_offset]);
    frame[mac_offset..].copy_from_slice(mac.as_bytes());
}
