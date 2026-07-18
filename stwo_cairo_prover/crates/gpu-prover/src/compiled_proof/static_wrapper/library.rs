//! Typed, address-free CUDA-library calls admitted by static wrapper manifests.

use super::CompiledProofError;

/// Logical A/B scratch-buffer identity. This is a direction selector, never a
/// device pointer or a process-local CUB selector.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum StaticCudaCubBuffer {
    A = 1,
    B = 2,
}

/// Exact identity of one stable ascending CUB u32 key/u32 value radix-sort.
///
/// `exact_temp_bytes` is the value returned by the admitted CUB build's sizing
/// query for `rows`; it is not the backing allocation's capacity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StaticCudaCubStableAscendingSortPairsU32V1 {
    word: u32,
    keys_from: StaticCudaCubBuffer,
    keys_to: StaticCudaCubBuffer,
    indices_from: StaticCudaCubBuffer,
    indices_to: StaticCudaCubBuffer,
    begin_bit: u8,
    end_bit: u8,
    rows: u32,
    exact_temp_bytes: u64,
}

impl StaticCudaCubStableAscendingSortPairsU32V1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        word: u32,
        keys_from: StaticCudaCubBuffer,
        keys_to: StaticCudaCubBuffer,
        indices_from: StaticCudaCubBuffer,
        indices_to: StaticCudaCubBuffer,
        begin_bit: u8,
        end_bit: u8,
        rows: u32,
        exact_temp_bytes: u64,
    ) -> Result<Self, CompiledProofError> {
        let call = Self {
            word,
            keys_from,
            keys_to,
            indices_from,
            indices_to,
            begin_bit,
            end_bit,
            rows,
            exact_temp_bytes,
        };
        if call.is_valid() {
            Ok(call)
        } else {
            Err(CompiledProofError::InvalidStaticWrapperManifest)
        }
    }

    pub const fn word(&self) -> u32 {
        self.word
    }

    pub const fn keys_from(&self) -> StaticCudaCubBuffer {
        self.keys_from
    }

    pub const fn keys_to(&self) -> StaticCudaCubBuffer {
        self.keys_to
    }

    pub const fn indices_from(&self) -> StaticCudaCubBuffer {
        self.indices_from
    }

    pub const fn indices_to(&self) -> StaticCudaCubBuffer {
        self.indices_to
    }

    pub const fn begin_bit(&self) -> u8 {
        self.begin_bit
    }

    pub const fn end_bit(&self) -> u8 {
        self.end_bit
    }

    pub const fn rows(&self) -> u32 {
        self.rows
    }

    pub const fn exact_temp_bytes(&self) -> u64 {
        self.exact_temp_bytes
    }

    const fn is_valid(&self) -> bool {
        self.keys_from as u8 != self.keys_to as u8
            && self.indices_from as u8 != self.indices_to as u8
            && self.begin_bit < self.end_bit
            && self.end_bit <= u32::BITS as u8
            && self.rows != 0
            && self.exact_temp_bytes != 0
    }
}

/// Exact identity of one out-of-place CUB u32 inclusive sum.
///
/// The wrapper contract fixes the input and output logical buffers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StaticCudaCubInclusiveSumU32V1 {
    rows: u32,
    exact_temp_bytes: u64,
}

impl StaticCudaCubInclusiveSumU32V1 {
    pub fn new(rows: u32, exact_temp_bytes: u64) -> Result<Self, CompiledProofError> {
        let call = Self {
            rows,
            exact_temp_bytes,
        };
        if call.is_valid() {
            Ok(call)
        } else {
            Err(CompiledProofError::InvalidStaticWrapperManifest)
        }
    }

    pub const fn rows(&self) -> u32 {
        self.rows
    }

    pub const fn exact_temp_bytes(&self) -> u64 {
        self.exact_temp_bytes
    }

    const fn is_valid(&self) -> bool {
        self.rows != 0 && self.exact_temp_bytes != 0
    }
}

/// Exact identity of one ordered asynchronous device-to-device copy.
///
/// Argument ordinals and byte offsets are relative to the enclosing static
/// wrapper ABI; they are never process-local addresses.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StaticCudaMemcpyD2DV1 {
    source_argument: u8,
    source_byte_offset: u64,
    destination_argument: u8,
    destination_byte_offset: u64,
    bytes: u64,
}

impl StaticCudaMemcpyD2DV1 {
    pub fn new(
        source_argument: u8,
        source_byte_offset: u64,
        destination_argument: u8,
        destination_byte_offset: u64,
        bytes: u64,
    ) -> Result<Self, CompiledProofError> {
        let copy = Self {
            source_argument,
            source_byte_offset,
            destination_argument,
            destination_byte_offset,
            bytes,
        };
        if copy.is_valid() {
            Ok(copy)
        } else {
            Err(CompiledProofError::InvalidStaticWrapperManifest)
        }
    }

    pub const fn source_argument(&self) -> u8 {
        self.source_argument
    }

    pub const fn source_byte_offset(&self) -> u64 {
        self.source_byte_offset
    }

    pub const fn destination_argument(&self) -> u8 {
        self.destination_argument
    }

    pub const fn destination_byte_offset(&self) -> u64 {
        self.destination_byte_offset
    }

    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    const fn is_valid(&self) -> bool {
        let Some(source_end) = self.source_byte_offset.checked_add(self.bytes) else {
            return false;
        };
        let Some(destination_end) = self.destination_byte_offset.checked_add(self.bytes) else {
            return false;
        };
        self.bytes != 0
            && (self.source_argument != self.destination_argument
                || source_end <= self.destination_byte_offset
                || destination_end <= self.source_byte_offset)
    }
}

/// Closed, versioned set of CUDA-library operations admitted into wrapper
/// identity. Each variant fixes its API, element types, ordering semantics,
/// stream ordering, and canonical field widths.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StaticCudaLibraryCallIdentity {
    CubStableAscendingSortPairsU32V1(StaticCudaCubStableAscendingSortPairsU32V1),
    CubInclusiveSumU32V1(StaticCudaCubInclusiveSumU32V1),
    MemcpyD2DV1(StaticCudaMemcpyD2DV1),
}

impl StaticCudaLibraryCallIdentity {
    pub const fn api(&self) -> &'static [u8] {
        match self {
            Self::CubStableAscendingSortPairsU32V1(_) => b"cub::DeviceRadixSort::SortPairs",
            Self::CubInclusiveSumU32V1(_) => b"cub::DeviceScan::InclusiveSum",
            Self::MemcpyD2DV1(_) => b"cudaMemcpyAsync(cudaMemcpyDeviceToDevice)",
        }
    }

    pub const fn library_managed_launch_geometry(&self) -> bool {
        true
    }

    pub const fn ordered_on_wrapper_stream(&self) -> bool {
        true
    }

    pub(super) const fn is_valid(&self) -> bool {
        match self {
            Self::CubStableAscendingSortPairsU32V1(call) => call.is_valid(),
            Self::CubInclusiveSumU32V1(call) => call.is_valid(),
            Self::MemcpyD2DV1(call) => call.is_valid(),
        }
    }

    pub(super) fn encode_into(&self, out: &mut Vec<u8>) {
        match self {
            Self::CubStableAscendingSortPairsU32V1(call) => {
                out.push(1);
                out.extend_from_slice(&call.word.to_le_bytes());
                out.push(call.keys_from as u8);
                out.push(call.keys_to as u8);
                out.push(call.indices_from as u8);
                out.push(call.indices_to as u8);
                out.push(call.begin_bit);
                out.push(call.end_bit);
                out.extend_from_slice(&call.rows.to_le_bytes());
                out.extend_from_slice(&call.exact_temp_bytes.to_le_bytes());
            }
            Self::CubInclusiveSumU32V1(call) => {
                out.push(2);
                out.extend_from_slice(&call.rows.to_le_bytes());
                out.extend_from_slice(&call.exact_temp_bytes.to_le_bytes());
            }
            Self::MemcpyD2DV1(call) => {
                out.push(3);
                out.push(call.source_argument);
                out.extend_from_slice(&call.source_byte_offset.to_le_bytes());
                out.push(call.destination_argument);
                out.extend_from_slice(&call.destination_byte_offset.to_le_bytes());
                out.extend_from_slice(&call.bytes.to_le_bytes());
            }
        }
    }
}
