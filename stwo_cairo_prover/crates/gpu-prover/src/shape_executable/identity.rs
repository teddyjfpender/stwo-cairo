use stwo_cairo_prover::witness::proof_shape::PendingRowsReason;

use super::*;

const TOPOLOGY_DOMAIN: &[u8] = b"stwo-cairo.topology-key.canonical.v1\0";
const WORKSPACE_DOMAIN: &[u8] = b"stwo-cairo.workspace-layout.canonical.v1\0";
#[cfg(test)]
const SHAPE_DOMAIN: &[u8] = b"stwo-cairo.shape-executable.canonical.v1\0";

/// Immutable, collision-resistant identity of a fully compiled shape.
/// Construction is impossible until real CompiledProof canonical bytes exist.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ShapeExecutableIdentity {
    canonical_encoding: Box<[u8]>,
    compiled_proof_encoding: Box<[u8]>,
    transcript_encoding: Box<[u8]>,
    digest: [u8; 32],
}

impl ShapeExecutableIdentity {
    pub fn canonical_encoding(&self) -> &[u8] {
        &self.canonical_encoding
    }

    pub const fn digest(&self) -> &[u8; 32] {
        &self.digest
    }

    pub fn compiled_proof_encoding(&self) -> &[u8] {
        &self.compiled_proof_encoding
    }

    pub fn transcript_encoding(&self) -> &[u8] {
        &self.transcript_encoding
    }
}

pub(super) fn encode_topology(key: &TopologyKey) -> Result<Vec<u8>, ShapeExecutableError> {
    let mut out = Encoder::new(TOPOLOGY_DOMAIN);
    out.u64(key.relation_graph_hash);
    out.count(key.shape.components().len())?;
    for component in key.shape.components() {
        out.bytes(component.id.as_bytes())?;
        match &component.rows {
            RowResolution::Absent => out.byte(0),
            RowResolution::Resolved(parts) => {
                out.byte(1);
                out.count(parts.len())?;
                for part in parts {
                    encode_part(&mut out, part.part);
                    out.u64(part.n_real_rows);
                    out.u64(part.padded_rows);
                }
            }
            RowResolution::Pending {
                reason,
                observed_n_real_rows,
            } => {
                out.byte(2);
                out.byte(pending_reason_tag(*reason));
                out.u64(*observed_n_real_rows);
            }
            RowResolution::Bounded { reason, bound } => {
                out.byte(3);
                out.byte(pending_reason_tag(*reason));
                out.u64(bound.observed_rows);
                out.u64(bound.max_rows);
                out.u64(bound.padded_capacity);
            }
        }
    }

    out.count(key.component_enable_bits.len())?;
    for &enabled in &key.component_enable_bits {
        out.byte(u8::from(enabled));
    }
    out.count(key.component_log_sizes.len())?;
    for &log_size in &key.component_log_sizes {
        out.u32(log_size);
    }
    out.count(key.claim_log_sizes.len())?;
    for tree in &key.claim_log_sizes {
        out.count(tree.len())?;
        for &log_size in tree {
            out.u32(log_size);
        }
    }
    out.u32(key.claim_public_data_felts);
    out.byte(preprocessed_variant_tag(key.preprocessed_trace_variant));
    out.count(key.preprocessed_columns.len())?;
    for (id, log_size) in &key.preprocessed_columns {
        out.bytes(id.as_bytes())?;
        out.u32(*log_size);
    }

    out.u32(key.pcs.pow_bits);
    out.u32(key.pcs.log_blowup_factor);
    out.u32(key.pcs.log_last_layer_degree_bound);
    out.size(key.pcs.n_queries)?;
    out.u32(key.pcs.fold_step);
    match key.pcs.lifting_log_size {
        Some(value) => {
            out.byte(1);
            out.u32(value);
        }
        None => out.byte(0),
    }
    out.byte(u8::from(key.include_all_preprocessed_columns));
    match key.execution_tables {
        Some(geometry) => {
            out.byte(1);
            out.size(geometry.n_addrs)?;
            out.size(geometry.n_big)?;
            out.size(geometry.n_small)?;
            out.size(geometry.public_memory_entries)?;
        }
        None => out.byte(0),
    }
    encode_policy(&mut out, key.policy)?;
    Ok(out.finish())
}

pub(super) fn encode_workspace(
    layout: &WorkspaceLayoutIdentity,
) -> Result<Vec<u8>, ShapeExecutableError> {
    let mut out = Encoder::new(WORKSPACE_DOMAIN);
    out.size(layout.total_words)?;
    out.count(layout.logical.len())?;
    for logical in &layout.logical {
        out.u32(logical.id.0);
        match logical.component {
            Some(component) => {
                out.byte(1);
                out.bytes(component.as_bytes())?;
            }
            None => out.byte(0),
        }
        match logical.part {
            Some(part) => {
                out.byte(1);
                encode_part(&mut out, part);
            }
            None => out.byte(0),
        }
        out.u32(logical.purpose as u32);
        out.u32(logical.ordinal);
        out.size(logical.len_words)?;
        out.byte(logical.lifetime.first as u8);
        out.byte(logical.lifetime.last as u8);
    }
    out.count(layout.bindings.len())?;
    for binding in &layout.bindings {
        out.u32(binding.logical.0);
        out.u32(binding.physical.0);
        out.size(binding.len_words)?;
    }
    out.count(layout.slots.len())?;
    for slot in &layout.slots {
        out.u32(slot.id.0);
        out.size(slot.offset_words)?;
        out.size(slot.len_words)?;
        out.size(slot.alignment_words)?;
    }
    Ok(out.finish())
}

/// Test-only assembly hook for exact-byte admission tests. Production has no
/// detached identity builder; the real emitter must eventually construct and
/// install `CompiledProof` plus this identity atomically.
#[cfg(test)]
pub(crate) fn compose_shape_identity(
    topology: &[u8],
    workspace: &[u8],
    transcript: &[u8],
    compiled_proof: &[u8],
) -> Result<ShapeExecutableIdentity, ShapeExecutableError> {
    if compiled_proof.is_empty() {
        return Err(ShapeExecutableError::MissingCompiledProofIdentity);
    }
    let mut out = Encoder::new(SHAPE_DOMAIN);
    out.section(b"topology", topology)?;
    out.section(b"workspace", workspace)?;
    out.section(b"transcript", transcript)?;
    out.section(b"compiled-proof", compiled_proof)?;
    let canonical_encoding = out.finish();
    Ok(ShapeExecutableIdentity {
        digest: *blake3::hash(&canonical_encoding).as_bytes(),
        canonical_encoding: canonical_encoding.into_boxed_slice(),
        compiled_proof_encoding: compiled_proof.into(),
        transcript_encoding: transcript.into(),
    })
}

fn encode_part(out: &mut Encoder, part: TracePartId) {
    match part {
        TracePartId::Main => out.byte(0),
        TracePartId::MemoryBig(index) => {
            out.byte(1);
            out.u32(index);
        }
        TracePartId::MemorySmall => out.byte(2),
    }
}

fn pending_reason_tag(reason: PendingRowsReason) -> u8 {
    match reason {
        PendingRowsReason::WitnessRelationFeeds => 0,
    }
}

fn encode_policy(
    out: &mut Encoder,
    policy: ProtocolPlanPolicy,
) -> Result<(), ShapeExecutableError> {
    out.byte(policy.resident_backend as u8);
    out.byte(policy.dynamic_commitment_leaf_schedule as u8);
    out.byte(policy.quotient_numerator_schedule as u8);
    out.u64(policy.channel_tag);
    out.u64(policy.kernel_manifest_hash);
    out.size(policy.composition_max_kernel_instrs)?;
    out.byte(policy.decommit_strategy as u8);
    out.size(policy.retained_lde_budget_bytes)?;
    out.size(policy.fixed_image_incremental_lde_budget_bytes)?;
    out.u32(policy.unretained_bottom_layers);
    out.u32(policy.max_fused_tail_levels);
    out.byte(policy.commit_mode as u8);
    out.byte(policy.direct_composition_retention_mode as u8);
    out.byte(policy.quotient_numerator_source_policy as u8);
    out.byte(policy.interpolation_mode as u8);
    out.byte(u8::from(policy.blake2s_interior_fused));
    out.byte(policy.composition_launch_mode as u8);
    out.byte(policy.relation_tail_mode as u8);
    out.byte(policy.fri_fold_launch_mode as u8);
    out.byte(policy.witness_feed_launch_mode as u8);
    Ok(())
}

struct Encoder(Vec<u8>);

impl Encoder {
    fn new(domain: &[u8]) -> Self {
        Self(domain.to_vec())
    }

    fn finish(self) -> Vec<u8> {
        self.0
    }

    fn raw(&mut self, bytes: &[u8]) {
        self.0.extend_from_slice(bytes);
    }

    fn byte(&mut self, value: u8) {
        self.0.push(value);
    }

    fn u32(&mut self, value: u32) {
        self.raw(&value.to_le_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.raw(&value.to_le_bytes());
    }

    fn size(&mut self, value: usize) -> Result<(), ShapeExecutableError> {
        let value = u64::try_from(value).map_err(|_| ShapeExecutableError::SizeOverflow)?;
        self.u64(value);
        Ok(())
    }

    fn count(&mut self, value: usize) -> Result<(), ShapeExecutableError> {
        self.size(value)
    }

    fn bytes(&mut self, bytes: &[u8]) -> Result<(), ShapeExecutableError> {
        self.count(bytes.len())?;
        self.raw(bytes);
        Ok(())
    }

    #[cfg(test)]
    fn section(&mut self, tag: &[u8], bytes: &[u8]) -> Result<(), ShapeExecutableError> {
        self.bytes(tag)?;
        self.bytes(bytes)
    }
}
