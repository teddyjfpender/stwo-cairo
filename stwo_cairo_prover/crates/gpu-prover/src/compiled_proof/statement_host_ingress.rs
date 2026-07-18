/// Shape-only class of an eager statement source.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StatementHostSourceKind {
    WitnessCasm,
}

/// Canonical address-free host payload encoding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StatementHostEncoding {
    RowMajorU32,
}

/// Inspectable Cairo trace part without borrowing producer-owned metadata.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StatementHostPart {
    Main,
    MemoryBig(u32),
    MemorySmall,
}

/// Exact shape of one eager host statement.
///
/// Payload bytes and host addresses are deliberately absent. The runtime
/// binds those separately against this source and its CASM contract identity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StatementHostSource {
    pub kind: StatementHostSourceKind,
    pub producer_ordinal: u32,
    pub component: Box<str>,
    pub part: StatementHostPart,
    pub encoding: StatementHostEncoding,
    pub words: usize,
    pub real_rows: usize,
    pub consumer_rows: usize,
    pub include_iota: bool,
    pub casm_contract_identity: [u8; 32],
}

impl StatementHostSource {
    pub(crate) fn has_valid_shape(&self) -> bool {
        !self.component.is_empty()
            && self.words != 0
            && self.real_rows != 0
            && self.consumer_rows >= self.real_rows
            && self.words <= u32::MAX as usize
            && self.real_rows <= u32::MAX as usize
            && self.consumer_rows <= u32::MAX as usize
            && self.casm_contract_identity != [0; 32]
    }
}
