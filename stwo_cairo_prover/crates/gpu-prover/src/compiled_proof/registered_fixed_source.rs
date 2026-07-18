use std::collections::BTreeSet;

use super::{CompiledProofError, ElementRange};

const REGISTERED_FIXED_SOURCE_DOMAIN: &[u8] =
    b"stwo-cairo.compiled-proof.registered-fixed-source.v1\0";

/// Address-free identity of one immutable process-owned fixed source.
///
/// `recipe_identity` names the canonical content recipe. Runtime admission
/// must additionally bind this authority to the checked registration's byte
/// digest and live column addresses before publishing a module global.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct RegisteredFixedSourceAuthority {
    recipe_identity: [u8; 32],
    source_rows: usize,
    padded_rows: usize,
    element_bytes: usize,
    columns: Box<[Box<[u8]>]>,
    canonical_encoding: Box<[u8]>,
    identity: [u8; 32],
}

impl RegisteredFixedSourceAuthority {
    pub fn new(
        recipe_identity: [u8; 32],
        source_rows: usize,
        padded_rows: usize,
        element_bytes: usize,
        columns: Vec<Vec<u8>>,
    ) -> Result<Self, CompiledProofError> {
        if recipe_identity == [0; 32]
            || source_rows == 0
            || padded_rows < source_rows
            || !padded_rows.is_power_of_two()
            || element_bytes == 0
            || columns.is_empty()
            || columns
                .iter()
                .any(|column| column.is_empty() || column.contains(&0))
            || columns.iter().collect::<BTreeSet<_>>().len() != columns.len()
        {
            return Err(CompiledProofError::InvalidRegisteredFixedSource);
        }
        let columns = columns
            .into_iter()
            .map(Vec::into_boxed_slice)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let canonical_encoding = encode(
            recipe_identity,
            source_rows,
            padded_rows,
            element_bytes,
            &columns,
        )?;
        let identity = digest(&canonical_encoding);
        Ok(Self {
            recipe_identity,
            source_rows,
            padded_rows,
            element_bytes,
            columns,
            canonical_encoding: canonical_encoding.into_boxed_slice(),
            identity,
        })
    }

    pub const fn recipe_identity(&self) -> &[u8; 32] {
        &self.recipe_identity
    }

    pub const fn source_rows(&self) -> usize {
        self.source_rows
    }

    pub const fn padded_rows(&self) -> usize {
        self.padded_rows
    }

    pub const fn element_bytes(&self) -> usize {
        self.element_bytes
    }

    pub fn columns(&self) -> impl ExactSizeIterator<Item = &[u8]> {
        self.columns.iter().map(Box::as_ref)
    }

    pub fn canonical_encoding(&self) -> &[u8] {
        &self.canonical_encoding
    }

    pub const fn identity(&self) -> &[u8; 32] {
        &self.identity
    }

    pub(crate) fn has_valid_identity(&self) -> Result<bool, CompiledProofError> {
        let canonical = encode(
            self.recipe_identity,
            self.source_rows,
            self.padded_rows,
            self.element_bytes,
            &self.columns,
        )?;
        Ok(canonical == self.canonical_encoding.as_ref() && self.identity == digest(&canonical))
    }
}

/// Exact immutable range read from one process-registered fixed-source column.
///
/// The authority is address-free. Runtime admission must resolve it through
/// the checked process registration rather than accepting a serialized device
/// pointer or a module-global relocation as a substitute.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct RegisteredFixedSourceRead {
    source: RegisteredFixedSourceAuthority,
    column: usize,
    elements: ElementRange,
}

impl RegisteredFixedSourceRead {
    pub fn new(
        source: RegisteredFixedSourceAuthority,
        column: usize,
        elements: ElementRange,
    ) -> Result<Self, CompiledProofError> {
        if !source.has_valid_identity()?
            || column >= source.columns().len()
            || elements.is_empty()
            || elements.end > source.padded_rows()
        {
            return Err(CompiledProofError::InvalidRegisteredFixedSourceRead);
        }
        Ok(Self {
            source,
            column,
            elements,
        })
    }

    pub const fn source(&self) -> &RegisteredFixedSourceAuthority {
        &self.source
    }

    pub const fn column(&self) -> usize {
        self.column
    }

    pub const fn elements(&self) -> ElementRange {
        self.elements
    }

    pub(crate) fn has_valid_identity(&self) -> Result<bool, CompiledProofError> {
        Ok(self.source.has_valid_identity()?
            && self.column < self.source.columns().len()
            && !self.elements.is_empty()
            && self.elements.end <= self.source.padded_rows())
    }
}

fn encode(
    recipe_identity: [u8; 32],
    source_rows: usize,
    padded_rows: usize,
    element_bytes: usize,
    columns: &[Box<[u8]>],
) -> Result<Vec<u8>, CompiledProofError> {
    let mut out = Vec::from(REGISTERED_FIXED_SOURCE_DOMAIN);
    out.extend_from_slice(&recipe_identity);
    push_size(&mut out, source_rows)?;
    push_size(&mut out, padded_rows)?;
    push_size(&mut out, element_bytes)?;
    push_size(&mut out, columns.len())?;
    for column in columns {
        push_bytes(&mut out, column)?;
    }
    Ok(out)
}

fn digest(canonical: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(REGISTERED_FIXED_SOURCE_DOMAIN);
    hasher.update(&(canonical.len() as u64).to_le_bytes());
    hasher.update(canonical);
    *hasher.finalize().as_bytes()
}

fn push_size(out: &mut Vec<u8>, value: usize) -> Result<(), CompiledProofError> {
    out.extend_from_slice(
        &u64::try_from(value)
            .map_err(|_| CompiledProofError::SizeOverflow)?
            .to_le_bytes(),
    );
    Ok(())
}

fn push_bytes(out: &mut Vec<u8>, bytes: &[u8]) -> Result<(), CompiledProofError> {
    push_size(out, bytes.len())?;
    out.extend_from_slice(bytes);
    Ok(())
}
