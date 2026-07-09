//! Fail-closed binding and replay for the arena-backed CUDA proof primitives.
//!
//! This module does not infer Cairo trace order. Commitment sources come from
//! the plan's canonical identities; generated relation layouts bind their own
//! trace/lookup columns. Only the quotient evaluation at the FRI seam remains a
//! typed caller binding until the resident quotient graph lands.

use core::ffi::c_void;
use std::collections::HashMap;

use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::SecureField;
use stwo::core::vcs::blake2_hash::Blake2sHash;
use stwo_backend_cuda::{
    ArenaError, ArenaSlice, CommitCoefficientColumn, CommitCoefficientGroup, CudaRuntimeError,
    DeviceTranscriptError, PreparedBlake2sTranscript, PreparedCommitError, PreparedCommitGraph,
    PreparedFriError, PreparedFriGraph, PreparedRelationGraph, RelationChallenges,
    RelationGraphError, RelationInstanceSources, RelationSourceLayout, TranscriptInputBinding,
    TranscriptInputId, TranscriptOutputBinding, TranscriptOutputId, TranscriptSegmentCursor,
    TranscriptSegmentStart,
};
use stwo_cairo_prover::witness::proof_shape::ProofShapeKey;

use crate::arena_plan::{
    BufferPurpose, CommitmentColumnSource, CommitmentTreeId, PlannedCommitment,
};
use crate::graphs::{GraphError, GraphSegment, GraphWorkspace, PhaseGraph};
use crate::relation::RelationTracePart;
use crate::relation_execution::RelationBatchKey;
use crate::transcript_plan::{
    CairoBlake2sTranscriptPlan, CairoTranscriptInput, CairoTranscriptOutput,
    CairoTranscriptSegment, TranscriptPlanError, TranscriptSegmentPlan,
};

/// Complete cache identity of one materialized resident workspace.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidentWorkspaceIdentity {
    pub shape_key: ProofShapeKey,
    pub protocol_key: u64,
    pub arena_base: usize,
}

impl ResidentWorkspaceIdentity {
    pub fn of(workspace: &GraphWorkspace) -> Self {
        Self {
            shape_key: workspace.plan().shape_key,
            protocol_key: workspace.plan().protocol_key,
            arena_base: workspace.arena().base_ptr().as_ptr() as usize,
        }
    }
}

/// The four-coordinate quotient evaluation consumed by prepared FRI.
#[derive(Clone, Copy, Debug)]
pub struct ResidentFriBinding {
    pub input_evaluation: ArenaSlice,
}

/// The quotient/FRI seam is the only source identity not yet carried by
/// `ProofArenaPlan`; every relation and commitment column is auto-bound.
pub struct ResidentSourceBindings {
    pub fri: ResidentFriBinding,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidentClaimedSum {
    pub batch: RelationBatchKey,
    pub instance_index: usize,
    pub value: SecureField,
}

/// The only interaction D2H boundary: one root plus four words per relation
/// instance, drained with a single stream synchronization.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InteractionTranscriptBoundary {
    pub root: Blake2sHash,
    pub claimed_sums: Vec<ResidentClaimedSum>,
}

#[derive(Debug)]
pub enum ResidentRuntimeError {
    WorkspaceIdentityMismatch {
        expected: ResidentWorkspaceIdentity,
        actual: ResidentWorkspaceIdentity,
    },
    SourceOutsideArena {
        slot: stwo_backend_cuda::ArenaSlotId,
    },
    MissingCommitmentSource {
        id: CommitmentTreeId,
        source: CommitmentColumnSource,
    },
    MissingRelationSource {
        batch: RelationBatchKey,
        instance_index: usize,
        ordinal: u32,
    },
    MissingPreparedCommitment(CommitmentTreeId),
    TranscriptScheduleMismatch {
        expected: u64,
        actual: u64,
    },
    TranscriptRequirementsMismatch,
    MissingTranscriptInput(TranscriptInputId),
    MissingTranscriptOutput(TranscriptOutputId),
    InvalidTranscriptSegment(usize),
    MissingTranscriptSegment(CairoTranscriptSegment),
    TranscriptBindingTooSmall {
        role: &'static str,
        required_words: usize,
        actual_words: usize,
    },
    TranscriptClaimWidthMismatch {
        expected_words: usize,
        actual_words: usize,
    },
    UnknownTranscriptRelationComponent(&'static str),
    StaleRelationChallenges,
    StaleFriChallenge(usize),
    FriRoundOutOfOrder {
        expected: usize,
        actual: usize,
    },
    FriRoundIndexTooLarge(usize),
    Arena(ArenaError),
    Cuda(CudaRuntimeError),
    Graph(GraphError),
    Commit(PreparedCommitError),
    Fri(PreparedFriError),
    Relation(RelationGraphError),
    DeviceTranscript(DeviceTranscriptError),
    TranscriptPlan(TranscriptPlanError),
}

impl core::fmt::Display for ResidentRuntimeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "resident CUDA runtime rejected proof binding: {self:?}")
    }
}

impl std::error::Error for ResidentRuntimeError {}

impl From<GraphError> for ResidentRuntimeError {
    fn from(value: GraphError) -> Self {
        Self::Graph(value)
    }
}

impl From<ArenaError> for ResidentRuntimeError {
    fn from(value: ArenaError) -> Self {
        Self::Arena(value)
    }
}

impl From<CudaRuntimeError> for ResidentRuntimeError {
    fn from(value: CudaRuntimeError) -> Self {
        Self::Cuda(value)
    }
}

impl From<PreparedCommitError> for ResidentRuntimeError {
    fn from(value: PreparedCommitError) -> Self {
        Self::Commit(value)
    }
}

impl From<PreparedFriError> for ResidentRuntimeError {
    fn from(value: PreparedFriError) -> Self {
        Self::Fri(value)
    }
}

impl From<RelationGraphError> for ResidentRuntimeError {
    fn from(value: RelationGraphError) -> Self {
        Self::Relation(value)
    }
}

impl From<DeviceTranscriptError> for ResidentRuntimeError {
    fn from(value: DeviceTranscriptError) -> Self {
        Self::DeviceTranscript(value)
    }
}

impl From<TranscriptPlanError> for ResidentRuntimeError {
    fn from(value: TranscriptPlanError) -> Self {
        Self::TranscriptPlan(value)
    }
}

#[derive(Debug)]
enum ResidentLaunchError {
    Commit(PreparedCommitError),
    Fri(PreparedFriError),
    Relation(RelationGraphError),
    Transcript(DeviceTranscriptError),
    Cuda(CudaRuntimeError),
    Binding(&'static str),
}

impl core::fmt::Display for ResidentLaunchError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Commit(error) => write!(f, "resident commitment launch rejected: {error}"),
            Self::Fri(error) => write!(f, "resident FRI launch rejected: {error}"),
            Self::Relation(error) => write!(f, "resident relation launch rejected: {error}"),
            Self::Transcript(error) => write!(f, "resident transcript launch rejected: {error}"),
            Self::Cuda(error) => write!(f, "resident CUDA handoff rejected: {error}"),
            Self::Binding(role) => write!(f, "resident transcript binding rejected: {role}"),
        }
    }
}

impl std::error::Error for ResidentLaunchError {}

/// Prepared proof primitives and their captured transcript-bounded subgraphs.
///
/// Captures are declared first so graph executables are destroyed before the
/// prepared launch descriptions. The workspace itself is borrowed, so its slab,
/// stream, and pool necessarily outlive every captured pointer.
pub struct ResidentGraphRuntime<'a> {
    captures: HashMap<GraphSegment, PhaseGraph>,
    commitments: Vec<(CommitmentTreeId, PreparedCommitGraph<'a>)>,
    relation: PreparedRelationGraph<'a>,
    interaction_claim_sources: Vec<ArenaSlice>,
    fri: PreparedFriGraph<'a>,
    transcript: PreparedBlake2sTranscript<'a>,
    transcript_inputs: Vec<(TranscriptInputId, ArenaSlice)>,
    transcript_outputs: Vec<(TranscriptOutputId, ArenaSlice)>,
    transcript_segments: Vec<TranscriptSegmentPlan>,
    transcript_cursor: TranscriptSegmentCursor,
    workspace: &'a GraphWorkspace,
    identity: ResidentWorkspaceIdentity,
    relation_challenge_generation: u64,
    launched_relation_challenge_generation: u64,
    fri_challenge_generations: Vec<u64>,
    launched_fri_challenge_generations: Vec<u64>,
    next_fri_round: Option<usize>,
}

impl<'a> ResidentGraphRuntime<'a> {
    /// Validate all identities and source order, upload immutable descriptor
    /// tables, and bind every launch to the workspace's isolated CUDA context.
    /// `setup_relation_challenges` initializes device storage only; interaction
    /// launch remains blocked until [`Self::upload_relation_challenges`] records
    /// the real post-base-commit transcript challenge.
    pub fn prepare(
        workspace: &'a GraphWorkspace,
        expected_identity: ResidentWorkspaceIdentity,
        bindings: ResidentSourceBindings,
        setup_relation_challenges: RelationChallenges<'_>,
        transcript_plan: &CairoBlake2sTranscriptPlan,
    ) -> Result<Self, ResidentRuntimeError> {
        let actual_identity = ResidentWorkspaceIdentity::of(workspace);
        if expected_identity != actual_identity {
            return Err(ResidentRuntimeError::WorkspaceIdentityMismatch {
                expected: expected_identity,
                actual: actual_identity,
            });
        }

        let relation_sources = arena_relation_sources(workspace)?;
        for source in relation_sources
            .iter()
            .flat_map(|instance| &instance.columns)
        {
            require_resident_source(workspace, *source)?;
        }
        require_resident_source(workspace, bindings.fri.input_evaluation)?;

        let arena = workspace.arena();
        let planned_transcript = workspace.plan().transcript();
        if planned_transcript.schedule_key != transcript_plan.schedule_key() {
            return Err(ResidentRuntimeError::TranscriptScheduleMismatch {
                expected: planned_transcript.schedule_key,
                actual: transcript_plan.schedule_key(),
            });
        }
        if planned_transcript.requirements != *transcript_plan.schedule().requirements() {
            return Err(ResidentRuntimeError::TranscriptRequirementsMismatch);
        }
        let (transcript_inputs, transcript_outputs) = transcript_bindings(workspace)?;
        let transcript = PreparedBlake2sTranscript::prepare(
            arena,
            transcript_plan.schedule().clone(),
            planned_transcript.slots,
            &transcript_inputs
                .iter()
                .map(|&(id, slice)| TranscriptInputBinding { id, slice })
                .collect::<Vec<_>>(),
            &transcript_outputs
                .iter()
                .map(|&(id, slice)| TranscriptOutputBinding { id, slice })
                .collect::<Vec<_>>(),
        )?;
        let transcript_cursor = transcript.segment_cursor();
        let relation_plan = workspace.plan().relation();
        let relation = PreparedRelationGraph::prepare(
            arena,
            relation_plan.execution.kernel_program(),
            &relation_plan.slots,
            &relation_sources,
            setup_relation_challenges,
        )?;
        let interaction_claim_sources = interaction_outputs_in_cairo_order(workspace, &relation)?;

        let mut commitments = Vec::with_capacity(workspace.plan().commitments().len());
        for planned in workspace.plan().commitments() {
            let groups = commitment_groups(workspace, planned)?;
            let twiddles = arena.bind(planned.twiddles.physical)?;
            commitments.push((
                planned.id,
                PreparedCommitGraph::prepare(
                    arena,
                    planned.config,
                    &groups,
                    twiddles,
                    &planned.slots,
                )?,
            ));
        }

        let fri_plan = workspace.plan().fri();
        let fri = PreparedFriGraph::prepare(
            arena,
            fri_plan.config,
            bindings.fri.input_evaluation,
            arena.bind(fri_plan.twiddles.physical)?,
            &fri_plan.slots,
        )?;
        let fri_rounds = fri.round_count();

        Ok(Self {
            captures: HashMap::new(),
            commitments,
            relation,
            interaction_claim_sources,
            fri,
            transcript,
            transcript_inputs,
            transcript_outputs,
            transcript_segments: transcript_plan.segments().to_vec(),
            transcript_cursor,
            workspace,
            identity: actual_identity,
            // Setup values only initialize the stable challenge slots. The
            // transcript boundary must explicitly publish generation one.
            relation_challenge_generation: 0,
            launched_relation_challenge_generation: 0,
            fri_challenge_generations: vec![0; fri_rounds],
            launched_fri_challenge_generations: vec![0; fri_rounds],
            next_fri_round: None,
        })
    }

    pub const fn identity(&self) -> ResidentWorkspaceIdentity {
        self.identity
    }

    pub fn begin_transcript_generation(
        &mut self,
        generation: u64,
    ) -> Result<(), ResidentRuntimeError> {
        self.transcript_cursor.begin_generation(generation)?;
        Ok(())
    }

    pub fn transcript_segment_count(&self) -> usize {
        self.transcript_segments.len()
    }

    pub fn launch_transcript_segment_eager(
        &mut self,
        segment_index: usize,
    ) -> Result<(), ResidentRuntimeError> {
        let segment = self.transcript_segments.get(segment_index).ok_or(
            ResidentRuntimeError::InvalidTranscriptSegment(segment_index),
        )?;
        let start = if segment_index == 0 {
            TranscriptSegmentStart::Initialize
        } else {
            TranscriptSegmentStart::Resume
        };
        let generation = self.transcript_cursor.generation();
        self.transcript.launch_segment(
            &mut self.transcript_cursor,
            generation,
            segment.operation_range.clone(),
            start,
        )?;
        Ok(())
    }

    /// Admit the corresponding already-captured transcript range immediately
    /// before graph replay. The same cursor rejects skipped, duplicated or stale
    /// transcript segments in eager and captured execution.
    pub fn admit_transcript_segment_replay(
        &mut self,
        segment_index: usize,
    ) -> Result<(), ResidentRuntimeError> {
        let segment = self.transcript_segments.get(segment_index).ok_or(
            ResidentRuntimeError::InvalidTranscriptSegment(segment_index),
        )?;
        let start = if segment_index == 0 {
            TranscriptSegmentStart::Initialize
        } else {
            TranscriptSegmentStart::Resume
        };
        let generation = self.transcript_cursor.generation();
        self.transcript_cursor.admit_segment(
            self.transcript.schedule(),
            generation,
            segment.operation_range.clone(),
            start,
        )?;
        Ok(())
    }

    pub fn require_transcript_complete(&self) -> Result<(), ResidentRuntimeError> {
        self.transcript_cursor.require_complete()?;
        Ok(())
    }

    pub fn transcript_input(
        &self,
        semantic: CairoTranscriptInput,
    ) -> Result<ArenaSlice, ResidentRuntimeError> {
        let id = semantic.id()?;
        self.transcript_inputs
            .iter()
            .find_map(|&(candidate, slice)| (candidate == id).then_some(slice))
            .ok_or(ResidentRuntimeError::MissingTranscriptInput(id))
    }

    pub fn transcript_output(
        &self,
        semantic: CairoTranscriptOutput,
    ) -> Result<ArenaSlice, ResidentRuntimeError> {
        let id = semantic.id()?;
        self.transcript_outputs
            .iter()
            .find_map(|&(candidate, slice)| (candidate == id).then_some(slice))
            .ok_or(ResidentRuntimeError::MissingTranscriptOutput(id))
    }

    /// Upload the compact host-owned transcript inputs permitted at ingest
    /// (salt, PCS parameters, public claim material and persistent roots) in one
    /// batch and one synchronization. Large proof data is never accepted here.
    pub fn upload_transcript_inputs_at_ingest(
        &self,
        inputs: &[(CairoTranscriptInput, Vec<u32>)],
    ) -> Result<(), ResidentRuntimeError> {
        let mut seen = Vec::with_capacity(inputs.len());
        let mut uploads = Vec::with_capacity(inputs.len());
        for (semantic, words) in inputs {
            let id = semantic.id()?;
            if seen.contains(&id) {
                return Err(ResidentRuntimeError::TranscriptRequirementsMismatch);
            }
            seen.push(id);
            let expected = self
                .transcript
                .schedule()
                .requirements()
                .inputs
                .iter()
                .find_map(|requirement| (requirement.id == id).then_some(requirement.min_words))
                .ok_or(ResidentRuntimeError::MissingTranscriptInput(id))?;
            if words.len() != expected {
                return Err(ResidentRuntimeError::TranscriptBindingTooSmall {
                    role: "ingest_transcript_input",
                    required_words: expected,
                    actual_words: words.len(),
                });
            }
            let destination = self.transcript_input(*semantic)?;
            uploads.push((destination, words.as_slice()));
        }
        for (destination, words) in uploads {
            // SAFETY: `words` remains borrowed through the one sync below and
            // the exact logical destination width was checked above.
            unsafe {
                self.workspace.arena().context().memcpy_h2d_async(
                    destination.as_void_ptr(),
                    words.as_ptr().cast(),
                    words.len() * core::mem::size_of::<u32>(),
                )?;
            }
        }
        self.workspace.arena().context().sync()?;
        Ok(())
    }

    /// Enqueue a device-to-device root handoff into the semantic transcript
    /// input slot. No root crosses PCIe and the copy is capture-safe.
    pub fn stage_commitment_root_for_transcript(
        &self,
        commitment: CommitmentTreeId,
        semantic: CairoTranscriptInput,
    ) -> Result<(), ResidentRuntimeError> {
        let source = self.commitment(commitment)?.root_slice();
        let destination = self.transcript_input(semantic)?;
        self.copy_transcript_words("commitment_root", source, destination, 8)
    }

    pub fn stage_fri_root_for_transcript(
        &self,
        tree_index: usize,
        layer_index: u32,
    ) -> Result<(), ResidentRuntimeError> {
        let source = self.fri.tree_root(tree_index)?;
        let destination = self.transcript_input(CairoTranscriptInput::FriLayerRoot(layer_index))?;
        self.copy_transcript_words("fri_root", source, destination, 8)
    }

    /// Gather relation claimed sums in the canonical Cairo claim order into the
    /// one contiguous transcript input. Each sum stays on device.
    pub fn stage_interaction_claim_for_transcript(&self) -> Result<(), ResidentRuntimeError> {
        let destination = self.transcript_input(CairoTranscriptInput::InteractionClaim)?;
        let required_words = self.interaction_claim_sources.len().checked_mul(4).ok_or(
            ResidentRuntimeError::TranscriptClaimWidthMismatch {
                expected_words: usize::MAX,
                actual_words: destination.len_words(),
            },
        )?;
        if destination.len_words() != required_words {
            return Err(ResidentRuntimeError::TranscriptClaimWidthMismatch {
                expected_words: required_words,
                actual_words: destination.len_words(),
            });
        }
        for (index, source) in self.interaction_claim_sources.iter().copied().enumerate() {
            let offset =
                index
                    .checked_mul(4)
                    .ok_or(ResidentRuntimeError::TranscriptClaimWidthMismatch {
                        expected_words: usize::MAX,
                        actual_words: destination.len_words(),
                    })?;
            // SAFETY: exact destination width was checked above; each source is
            // a four-word claimed sum owned by the same arena/context.
            unsafe {
                self.workspace.arena().context().memcpy_d2d_async(
                    destination.as_u32_ptr().add(offset).cast(),
                    source.as_void_ptr().cast_const(),
                    4 * core::mem::size_of::<u32>(),
                )?;
            }
        }
        Ok(())
    }

    /// Make the device channel's `[z, alpha]` output authoritative for relation
    /// execution without reconstructing CommonLookupElements on the host.
    pub fn publish_relation_challenges_from_transcript(
        &mut self,
    ) -> Result<(), ResidentRuntimeError> {
        let drawn = self.transcript_output(CairoTranscriptOutput::CommonLookupElements)?;
        self.relation.expand_challenges_from_transcript(drawn)?;
        self.relation_challenge_generation = self
            .relation_challenge_generation
            .checked_add(1)
            .ok_or(ResidentRuntimeError::StaleRelationChallenges)?;
        Ok(())
    }

    pub fn publish_fri_challenge_from_transcript(
        &mut self,
        round_index: usize,
    ) -> Result<(), ResidentRuntimeError> {
        let layer = u32::try_from(round_index)
            .map_err(|_| ResidentRuntimeError::FriRoundIndexTooLarge(round_index))?;
        let source = self.transcript_output(CairoTranscriptOutput::FriFoldingChallenge(layer))?;
        let destination = self.fri_round_challenge_destination(round_index)?;
        self.copy_transcript_words("fri_challenge", source, destination, 4)?;
        self.mark_fri_round_challenge_ready(round_index)
    }

    pub fn capture_base_commit_only(&mut self) -> Result<(), ResidentRuntimeError> {
        let commitment_index = self
            .commitments
            .iter()
            .position(|(id, _)| *id == CommitmentTreeId::Base)
            .ok_or(ResidentRuntimeError::MissingPreparedCommitment(
                CommitmentTreeId::Base,
            ))?;
        let transcript_segment =
            self.transcript_segment_index(CairoTranscriptSegment::BootstrapAndLookup)?;
        let range = self.transcript_segments[transcript_segment]
            .operation_range
            .clone();
        let root_destination = self.transcript_input(CairoTranscriptInput::BaseRoot)?;
        let lookup_output = self.transcript_output(CairoTranscriptOutput::CommonLookupElements)?;
        let commitment = &self.commitments[commitment_index].1;
        let root_source = commitment.root_slice();
        let transcript = &self.transcript;
        let relation = &self.relation;
        let cursor = &mut self.transcript_cursor;
        let generation = cursor.generation();
        let graph = PhaseGraph::capture(
            self.workspace.key(GraphSegment::IngestWitnessBaseCommit),
            self.workspace.arena(),
            |arena| {
                commitment.launch().map_err(ResidentLaunchError::Commit)?;
                enqueue_copy_words(arena, root_source, root_destination, 8)?;
                transcript
                    .launch_segment(
                        cursor,
                        generation,
                        range,
                        TranscriptSegmentStart::Initialize,
                    )
                    .map_err(ResidentLaunchError::Transcript)?;
                relation
                    .expand_challenges_from_transcript(lookup_output)
                    .map_err(ResidentLaunchError::Relation)
            },
        )?;
        self.captures
            .insert(GraphSegment::IngestWitnessBaseCommit, graph);
        Ok(())
    }

    pub fn capture_interaction_relation_and_commit(&mut self) -> Result<(), ResidentRuntimeError> {
        let commitment_index = self
            .commitments
            .iter()
            .position(|(id, _)| *id == CommitmentTreeId::Interaction)
            .ok_or(ResidentRuntimeError::MissingPreparedCommitment(
                CommitmentTreeId::Interaction,
            ))?;
        let transcript_segment =
            self.transcript_segment_index(CairoTranscriptSegment::InteractionAndComposition)?;
        let range = self.transcript_segments[transcript_segment]
            .operation_range
            .clone();
        let claim_destination = self.transcript_input(CairoTranscriptInput::InteractionClaim)?;
        let root_destination = self.transcript_input(CairoTranscriptInput::InteractionRoot)?;
        let claim_sources = &self.interaction_claim_sources;
        let relation = &self.relation;
        let commitment = &self.commitments[commitment_index].1;
        let root_source = commitment.root_slice();
        let transcript = &self.transcript;
        let cursor = &mut self.transcript_cursor;
        let generation = cursor.generation();
        let graph = PhaseGraph::capture(
            self.workspace.key(GraphSegment::InteractionCommit),
            self.workspace.arena(),
            |arena| {
                relation.launch().map_err(ResidentLaunchError::Relation)?;
                commitment.launch().map_err(ResidentLaunchError::Commit)?;
                enqueue_claimed_sums(arena, claim_sources, claim_destination)?;
                enqueue_copy_words(arena, root_source, root_destination, 8)?;
                transcript
                    .launch_segment(cursor, generation, range, TranscriptSegmentStart::Resume)
                    .map_err(ResidentLaunchError::Transcript)
            },
        )?;
        self.captures.insert(GraphSegment::InteractionCommit, graph);
        Ok(())
    }

    pub fn capture_composition_commit_only(&mut self) -> Result<(), ResidentRuntimeError> {
        let commitment_index = self
            .commitments
            .iter()
            .position(|(id, _)| *id == CommitmentTreeId::Composition)
            .ok_or(ResidentRuntimeError::MissingPreparedCommitment(
                CommitmentTreeId::Composition,
            ))?;
        let transcript_segment =
            self.transcript_segment_index(CairoTranscriptSegment::CompositionAndOods)?;
        let range = self.transcript_segments[transcript_segment]
            .operation_range
            .clone();
        let root_destination = self.transcript_input(CairoTranscriptInput::CompositionRoot)?;
        let commitment = &self.commitments[commitment_index].1;
        let root_source = commitment.root_slice();
        let transcript = &self.transcript;
        let cursor = &mut self.transcript_cursor;
        let generation = cursor.generation();
        let graph = PhaseGraph::capture(
            self.workspace.key(GraphSegment::CompositionQuotientCommit),
            self.workspace.arena(),
            |arena| {
                commitment.launch().map_err(ResidentLaunchError::Commit)?;
                enqueue_copy_words(arena, root_source, root_destination, 8)?;
                transcript
                    .launch_segment(cursor, generation, range, TranscriptSegmentStart::Resume)
                    .map_err(ResidentLaunchError::Transcript)
            },
        )?;
        self.captures
            .insert(GraphSegment::CompositionQuotientCommit, graph);
        Ok(())
    }

    /// Capture the OODS-value absorb and quotient challenge boundary. The OODS
    /// evaluator is bound into this segment once its prepared graph is installed;
    /// today the exact device input slot is already stable and mandatory.
    pub fn capture_oods_transcript_boundary(&mut self) -> Result<(), ResidentRuntimeError> {
        let transcript_segment =
            self.transcript_segment_index(CairoTranscriptSegment::OodsAndQuotient)?;
        let range = self.transcript_segments[transcript_segment]
            .operation_range
            .clone();
        let transcript = &self.transcript;
        let cursor = &mut self.transcript_cursor;
        let generation = cursor.generation();
        let graph = PhaseGraph::capture(
            self.workspace.key(GraphSegment::OodsEvaluation),
            self.workspace.arena(),
            |_| {
                transcript
                    .launch_segment(cursor, generation, range, TranscriptSegmentStart::Resume)
                    .map_err(ResidentLaunchError::Transcript)
            },
        )?;
        self.captures.insert(GraphSegment::OodsEvaluation, graph);
        Ok(())
    }

    pub fn capture_fri_first_tree(&mut self) -> Result<(), ResidentRuntimeError> {
        let transcript_segment =
            self.transcript_segment_index(CairoTranscriptSegment::FriLayer(0))?;
        let range = self.transcript_segments[transcript_segment]
            .operation_range
            .clone();
        let root_source = self.fri.tree_root(0)?;
        let root_destination = self.transcript_input(CairoTranscriptInput::FriLayerRoot(0))?;
        let challenge_source =
            self.transcript_output(CairoTranscriptOutput::FriFoldingChallenge(0))?;
        let challenge_destination = self.fri.round_challenge_slice(0)?;
        let fri = &self.fri;
        let transcript = &self.transcript;
        let cursor = &mut self.transcript_cursor;
        let generation = cursor.generation();
        let graph = PhaseGraph::capture(
            self.workspace.key(GraphSegment::FriLayer(0)),
            self.workspace.arena(),
            |arena| {
                fri.launch_first_tree().map_err(ResidentLaunchError::Fri)?;
                enqueue_copy_words(arena, root_source, root_destination, 8)?;
                transcript
                    .launch_segment(cursor, generation, range, TranscriptSegmentStart::Resume)
                    .map_err(ResidentLaunchError::Transcript)?;
                enqueue_copy_words(arena, challenge_source, challenge_destination, 4)
            },
        )?;
        self.captures.insert(GraphSegment::FriLayer(0), graph);
        Ok(())
    }

    pub fn capture_fri_round(&mut self, round_index: usize) -> Result<(), ResidentRuntimeError> {
        // Validate the round before beginning stream capture.
        let _ = self.fri.round_challenge_slice(round_index)?;
        let segment = fri_round_segment(round_index)?;
        let output_tree = self
            .fri
            .requirements()
            .rounds
            .get(round_index)
            .ok_or_else(|| PreparedFriError::InvalidRoundIndex(round_index))?
            .output_tree;
        let transcript_tail = output_tree
            .map(|tree_index| {
                let layer = u32::try_from(tree_index)
                    .map_err(|_| ResidentRuntimeError::FriRoundIndexTooLarge(tree_index))?;
                let transcript_segment =
                    self.transcript_segment_index(CairoTranscriptSegment::FriLayer(layer))?;
                Ok::<_, ResidentRuntimeError>((
                    self.transcript_segments[transcript_segment]
                        .operation_range
                        .clone(),
                    self.fri.tree_root(tree_index)?,
                    self.transcript_input(CairoTranscriptInput::FriLayerRoot(layer))?,
                    self.transcript_output(CairoTranscriptOutput::FriFoldingChallenge(layer))?,
                    self.fri.round_challenge_slice(tree_index)?,
                ))
            })
            .transpose()?;
        let fri = &self.fri;
        let transcript = &self.transcript;
        let cursor = &mut self.transcript_cursor;
        let generation = cursor.generation();
        let graph = PhaseGraph::capture(
            self.workspace.key(segment),
            self.workspace.arena(),
            |arena| {
                fri.launch_round(round_index)
                    .map(|_| ())
                    .map_err(ResidentLaunchError::Fri)?;
                if let Some((
                    range,
                    root_source,
                    root_destination,
                    challenge_source,
                    challenge_destination,
                )) = transcript_tail
                {
                    enqueue_copy_words(arena, root_source, root_destination, 8)?;
                    transcript
                        .launch_segment(cursor, generation, range, TranscriptSegmentStart::Resume)
                        .map_err(ResidentLaunchError::Transcript)?;
                    enqueue_copy_words(arena, challenge_source, challenge_destination, 4)?;
                }
                Ok(())
            },
        )?;
        self.captures.insert(segment, graph);
        Ok(())
    }

    pub fn capture_final_transcript_boundary(&mut self) -> Result<(), ResidentRuntimeError> {
        let transcript_segment =
            self.transcript_segment_index(CairoTranscriptSegment::FriLastLayerAndQueries)?;
        let range = self.transcript_segments[transcript_segment]
            .operation_range
            .clone();
        let transcript = &self.transcript;
        let query_source = self.transcript_output(CairoTranscriptOutput::QueryPositions)?;
        let (query_destination, query_words) = self.query_indices_destination()?;
        let cursor = &mut self.transcript_cursor;
        let generation = cursor.generation();
        let graph = PhaseGraph::capture(
            self.workspace
                .key(GraphSegment::OodsQueriesDecommitAssemble),
            self.workspace.arena(),
            |arena| {
                transcript
                    .launch_segment(cursor, generation, range, TranscriptSegmentStart::Resume)
                    .map_err(ResidentLaunchError::Transcript)?;
                enqueue_copy_words(arena, query_source, query_destination, query_words)
            },
        )?;
        self.captures
            .insert(GraphSegment::OodsQueriesDecommitAssemble, graph);
        Ok(())
    }

    pub fn capture_all_prepared_subgraphs(&mut self) -> Result<(), ResidentRuntimeError> {
        self.begin_transcript_generation(1)?;
        self.capture_base_commit_only()?;
        self.capture_interaction_relation_and_commit()?;
        self.capture_composition_commit_only()?;
        self.capture_oods_transcript_boundary()?;
        self.capture_fri_first_tree()?;
        for round in 0..self.fri.round_count() {
            self.capture_fri_round(round)?;
        }
        self.capture_final_transcript_boundary()?;
        self.require_transcript_complete()?;
        Ok(())
    }

    pub fn replay_base_commit_only(&mut self) -> Result<(), ResidentRuntimeError> {
        let segment = self.transcript_segment_index(CairoTranscriptSegment::BootstrapAndLookup)?;
        self.admit_transcript_segment_replay(segment)?;
        self.replay(GraphSegment::IngestWitnessBaseCommit)?;
        self.relation_challenge_generation = self
            .relation_challenge_generation
            .checked_add(1)
            .ok_or(ResidentRuntimeError::StaleRelationChallenges)?;
        Ok(())
    }

    pub fn replay_interaction_relation_and_commit(&mut self) -> Result<(), ResidentRuntimeError> {
        if self.relation_challenge_generation <= self.launched_relation_challenge_generation {
            return Err(ResidentRuntimeError::StaleRelationChallenges);
        }
        let transcript_segment =
            self.transcript_segment_index(CairoTranscriptSegment::InteractionAndComposition)?;
        self.admit_transcript_segment_replay(transcript_segment)?;
        self.replay(GraphSegment::InteractionCommit)?;
        self.launched_relation_challenge_generation = self.relation_challenge_generation;
        Ok(())
    }

    pub fn replay_composition_commit_only(&mut self) -> Result<(), ResidentRuntimeError> {
        let transcript_segment =
            self.transcript_segment_index(CairoTranscriptSegment::CompositionAndOods)?;
        self.admit_transcript_segment_replay(transcript_segment)?;
        self.replay(GraphSegment::CompositionQuotientCommit)
    }

    pub fn replay_oods_transcript_boundary(&mut self) -> Result<(), ResidentRuntimeError> {
        let transcript_segment =
            self.transcript_segment_index(CairoTranscriptSegment::OodsAndQuotient)?;
        self.admit_transcript_segment_replay(transcript_segment)?;
        self.replay(GraphSegment::OodsEvaluation)
    }

    pub fn replay_fri_first_tree(&mut self) -> Result<(), ResidentRuntimeError> {
        let transcript_segment =
            self.transcript_segment_index(CairoTranscriptSegment::FriLayer(0))?;
        self.admit_transcript_segment_replay(transcript_segment)?;
        self.replay(GraphSegment::FriLayer(0))?;
        self.mark_fri_round_challenge_ready(0)?;
        self.next_fri_round = Some(0);
        Ok(())
    }

    pub fn replay_fri_round(&mut self, round_index: usize) -> Result<(), ResidentRuntimeError> {
        self.require_next_fri_round(round_index)?;
        let generation = *self
            .fri_challenge_generations
            .get(round_index)
            .ok_or_else(|| PreparedFriError::InvalidRoundIndex(round_index))?;
        let launched = self.launched_fri_challenge_generations[round_index];
        if generation <= launched {
            return Err(ResidentRuntimeError::StaleFriChallenge(round_index));
        }
        let output_tree = self
            .fri
            .requirements()
            .rounds
            .get(round_index)
            .ok_or_else(|| PreparedFriError::InvalidRoundIndex(round_index))?
            .output_tree;
        if let Some(tree_index) = output_tree {
            let layer = u32::try_from(tree_index)
                .map_err(|_| ResidentRuntimeError::FriRoundIndexTooLarge(tree_index))?;
            let transcript_segment =
                self.transcript_segment_index(CairoTranscriptSegment::FriLayer(layer))?;
            self.admit_transcript_segment_replay(transcript_segment)?;
        }
        self.replay(fri_round_segment(round_index)?)?;
        self.launched_fri_challenge_generations[round_index] = generation;
        if let Some(tree_index) = output_tree {
            self.mark_fri_round_challenge_ready(tree_index)?;
        }
        self.next_fri_round = Some(round_index + 1);
        Ok(())
    }

    pub fn replay_final_transcript_boundary(&mut self) -> Result<(), ResidentRuntimeError> {
        let transcript_segment =
            self.transcript_segment_index(CairoTranscriptSegment::FriLastLayerAndQueries)?;
        self.admit_transcript_segment_replay(transcript_segment)?;
        self.replay(GraphSegment::OodsQueriesDecommitAssemble)?;
        self.require_transcript_complete()
    }

    /// Host-channel migration path. Device transcript mode writes directly to
    /// [`Self::fri_round_challenge_destination`] then calls
    /// [`Self::mark_fri_round_challenge_ready`].
    pub fn upload_fri_round_challenge(
        &mut self,
        round_index: usize,
        value: SecureField,
    ) -> Result<(), ResidentRuntimeError> {
        self.fri
            .upload_round_challenge_at_transcript_boundary(round_index, value)?;
        self.mark_fri_round_challenge_ready(round_index)
    }

    pub fn fri_round_challenge_destination(
        &self,
        round_index: usize,
    ) -> Result<ArenaSlice, ResidentRuntimeError> {
        Ok(self.fri.round_challenge_slice(round_index)?)
    }

    pub fn mark_fri_round_challenge_ready(
        &mut self,
        round_index: usize,
    ) -> Result<(), ResidentRuntimeError> {
        let generation = self
            .fri_challenge_generations
            .get_mut(round_index)
            .ok_or_else(|| PreparedFriError::InvalidRoundIndex(round_index))?;
        *generation = generation
            .checked_add(1)
            .ok_or(ResidentRuntimeError::FriRoundIndexTooLarge(round_index))?;
        Ok(())
    }

    pub fn upload_relation_challenges(
        &mut self,
        challenges: RelationChallenges<'_>,
    ) -> Result<(), ResidentRuntimeError> {
        self.relation
            .upload_challenges_at_transcript_boundary(challenges)?;
        self.relation_challenge_generation = self
            .relation_challenge_generation
            .checked_add(1)
            .ok_or(ResidentRuntimeError::StaleRelationChallenges)?;
        Ok(())
    }

    pub fn launch_base_commit_eager(&mut self) -> Result<(), ResidentRuntimeError> {
        self.commitment(CommitmentTreeId::Base)?.launch()?;
        self.stage_commitment_root_for_transcript(
            CommitmentTreeId::Base,
            CairoTranscriptInput::BaseRoot,
        )?;
        let segment = self.transcript_segment_index(CairoTranscriptSegment::BootstrapAndLookup)?;
        self.launch_transcript_segment_eager(segment)?;
        self.publish_relation_challenges_from_transcript()?;
        Ok(())
    }

    pub fn launch_interaction_eager(&mut self) -> Result<(), ResidentRuntimeError> {
        if self.relation_challenge_generation <= self.launched_relation_challenge_generation {
            return Err(ResidentRuntimeError::StaleRelationChallenges);
        }
        self.relation.launch()?;
        self.commitment(CommitmentTreeId::Interaction)?.launch()?;
        self.stage_interaction_claim_for_transcript()?;
        self.stage_commitment_root_for_transcript(
            CommitmentTreeId::Interaction,
            CairoTranscriptInput::InteractionRoot,
        )?;
        let transcript_segment =
            self.transcript_segment_index(CairoTranscriptSegment::InteractionAndComposition)?;
        self.launch_transcript_segment_eager(transcript_segment)?;
        self.launched_relation_challenge_generation = self.relation_challenge_generation;
        Ok(())
    }

    pub fn launch_composition_commit_eager(&mut self) -> Result<(), ResidentRuntimeError> {
        self.commitment(CommitmentTreeId::Composition)?.launch()?;
        self.stage_commitment_root_for_transcript(
            CommitmentTreeId::Composition,
            CairoTranscriptInput::CompositionRoot,
        )?;
        let transcript_segment =
            self.transcript_segment_index(CairoTranscriptSegment::CompositionAndOods)?;
        self.launch_transcript_segment_eager(transcript_segment)?;
        Ok(())
    }

    pub fn launch_oods_transcript_boundary_eager(&mut self) -> Result<(), ResidentRuntimeError> {
        let transcript_segment =
            self.transcript_segment_index(CairoTranscriptSegment::OodsAndQuotient)?;
        self.launch_transcript_segment_eager(transcript_segment)
    }

    pub fn launch_fri_first_tree_eager(&mut self) -> Result<(), ResidentRuntimeError> {
        self.fri.launch_first_tree()?;
        self.stage_fri_root_for_transcript(0, 0)?;
        let transcript_segment =
            self.transcript_segment_index(CairoTranscriptSegment::FriLayer(0))?;
        self.launch_transcript_segment_eager(transcript_segment)?;
        self.publish_fri_challenge_from_transcript(0)?;
        self.next_fri_round = Some(0);
        Ok(())
    }

    pub fn launch_fri_round_eager(
        &mut self,
        round_index: usize,
    ) -> Result<Option<usize>, ResidentRuntimeError> {
        self.require_next_fri_round(round_index)?;
        let generation = *self
            .fri_challenge_generations
            .get(round_index)
            .ok_or_else(|| PreparedFriError::InvalidRoundIndex(round_index))?;
        let launched = &mut self.launched_fri_challenge_generations[round_index];
        if generation <= *launched {
            return Err(ResidentRuntimeError::StaleFriChallenge(round_index));
        }
        let tree = self.fri.launch_round(round_index)?;
        *launched = generation;
        if let Some(tree_index) = tree {
            let layer = u32::try_from(tree_index)
                .map_err(|_| ResidentRuntimeError::FriRoundIndexTooLarge(tree_index))?;
            self.stage_fri_root_for_transcript(tree_index, layer)?;
            let transcript_segment =
                self.transcript_segment_index(CairoTranscriptSegment::FriLayer(layer))?;
            self.launch_transcript_segment_eager(transcript_segment)?;
            self.publish_fri_challenge_from_transcript(tree_index)?;
        }
        self.next_fri_round = Some(round_index + 1);
        Ok(tree)
    }

    pub fn launch_final_transcript_boundary_eager(&mut self) -> Result<(), ResidentRuntimeError> {
        let transcript_segment =
            self.transcript_segment_index(CairoTranscriptSegment::FriLastLayerAndQueries)?;
        self.launch_transcript_segment_eager(transcript_segment)?;
        let query_source = self.transcript_output(CairoTranscriptOutput::QueryPositions)?;
        let (query_destination, query_words) = self.query_indices_destination()?;
        self.copy_transcript_words(
            "query_positions",
            query_source,
            query_destination,
            query_words,
        )?;
        self.require_transcript_complete()
    }

    pub fn read_commitment_root(
        &self,
        id: CommitmentTreeId,
    ) -> Result<Blake2sHash, ResidentRuntimeError> {
        Ok(self.commitment(id)?.read_root_at_transcript_boundary()?)
    }

    pub fn read_interaction_transcript_boundary(
        &self,
    ) -> Result<InteractionTranscriptBoundary, ResidentRuntimeError> {
        let interaction = self.commitment(CommitmentTreeId::Interaction)?;
        let outputs: Vec<_> = self.relation.outputs().collect();
        let mut words = vec![[0u32; 4]; outputs.len()];
        let mut root = Blake2sHash::default();
        for (output, destination) in outputs.iter().zip(&mut words) {
            unsafe {
                self.workspace.arena().context().memcpy_d2h_async(
                    destination.as_mut_ptr().cast(),
                    output.claimed_sum.as_void_ptr().cast_const(),
                    core::mem::size_of_val(destination),
                )?;
            }
        }
        unsafe {
            self.workspace.arena().context().memcpy_d2h_async(
                root.0.as_mut_ptr().cast::<c_void>(),
                interaction.root_slice().as_void_ptr().cast_const(),
                core::mem::size_of::<Blake2sHash>(),
            )?;
        }
        self.workspace.arena().context().sync()?;

        let claimed_sums = outputs
            .into_iter()
            .zip(words)
            .map(|(output, coordinates)| ResidentClaimedSum {
                batch: self.workspace.plan().relation().execution.batches[output.batch_index],
                instance_index: output.instance_index,
                value: SecureField::from_m31_array(coordinates.map(M31::from_u32_unchecked)),
            })
            .collect();
        Ok(InteractionTranscriptBoundary { root, claimed_sums })
    }

    pub fn read_fri_tree_root(
        &self,
        tree_index: usize,
    ) -> Result<Blake2sHash, ResidentRuntimeError> {
        Ok(self.fri.read_tree_root(tree_index)?)
    }

    pub fn fri_round_output_tree(
        &self,
        round_index: usize,
    ) -> Result<Option<usize>, ResidentRuntimeError> {
        Ok(self
            .fri
            .requirements()
            .rounds
            .get(round_index)
            .ok_or_else(|| PreparedFriError::InvalidRoundIndex(round_index))?
            .output_tree)
    }

    fn commitment(
        &self,
        id: CommitmentTreeId,
    ) -> Result<&PreparedCommitGraph<'a>, ResidentRuntimeError> {
        self.commitments
            .iter()
            .find_map(|(candidate, graph)| (*candidate == id).then_some(graph))
            .ok_or(ResidentRuntimeError::MissingPreparedCommitment(id))
    }

    fn transcript_segment_index(
        &self,
        semantic: CairoTranscriptSegment,
    ) -> Result<usize, ResidentRuntimeError> {
        self.transcript_segments
            .iter()
            .position(|segment| segment.segment == semantic)
            .ok_or(ResidentRuntimeError::MissingTranscriptSegment(semantic))
    }

    fn query_indices_destination(&self) -> Result<(ArenaSlice, usize), ResidentRuntimeError> {
        let (logical, _) = self
            .workspace
            .plan()
            .find(None, None, BufferPurpose::QueryIndices, 0)
            .ok_or(ResidentRuntimeError::TranscriptRequirementsMismatch)?;
        Ok(self.workspace.bind(logical.id)?)
    }

    fn copy_transcript_words(
        &self,
        role: &'static str,
        source: ArenaSlice,
        destination: ArenaSlice,
        words: usize,
    ) -> Result<(), ResidentRuntimeError> {
        if source.len_words() < words {
            return Err(ResidentRuntimeError::TranscriptBindingTooSmall {
                role,
                required_words: words,
                actual_words: source.len_words(),
            });
        }
        if destination.len_words() < words {
            return Err(ResidentRuntimeError::TranscriptBindingTooSmall {
                role,
                required_words: words,
                actual_words: destination.len_words(),
            });
        }
        let bytes = words.checked_mul(core::mem::size_of::<u32>()).ok_or(
            ResidentRuntimeError::TranscriptBindingTooSmall {
                role,
                required_words: words,
                actual_words: destination.len_words(),
            },
        )?;
        // SAFETY: both slices are arena-owned and the word capacities above
        // cover the exact non-overlapping semantic handoff.
        unsafe {
            self.workspace.arena().context().memcpy_d2d_async(
                destination.as_void_ptr(),
                source.as_void_ptr().cast_const(),
                bytes,
            )?;
        }
        Ok(())
    }

    fn replay(&self, segment: GraphSegment) -> Result<(), ResidentRuntimeError> {
        let graph = self
            .captures
            .get(&segment)
            .ok_or(GraphError::MissingSegment(segment))?;
        graph.replay(self.workspace.arena())?;
        Ok(())
    }

    fn require_next_fri_round(&self, round_index: usize) -> Result<(), ResidentRuntimeError> {
        let expected = self.next_fri_round.unwrap_or(0);
        if self.next_fri_round.is_none() || round_index != expected {
            return Err(ResidentRuntimeError::FriRoundOutOfOrder {
                expected,
                actual: round_index,
            });
        }
        Ok(())
    }
}

fn enqueue_copy_words(
    arena: &stwo_backend_cuda::DeviceArena,
    source: ArenaSlice,
    destination: ArenaSlice,
    words: usize,
) -> Result<(), ResidentLaunchError> {
    if source.len_words() < words || destination.len_words() < words {
        return Err(ResidentLaunchError::Binding("device word copy capacity"));
    }
    let bytes = words
        .checked_mul(core::mem::size_of::<u32>())
        .ok_or(ResidentLaunchError::Binding("device word copy overflow"))?;
    // SAFETY: both arena ranges were capacity-checked and semantic producers
    // and consumers are distinct live slots.
    unsafe {
        arena
            .context()
            .memcpy_d2d_async(
                destination.as_void_ptr(),
                source.as_void_ptr().cast_const(),
                bytes,
            )
            .map_err(ResidentLaunchError::Cuda)?;
    }
    Ok(())
}

fn enqueue_claimed_sums(
    arena: &stwo_backend_cuda::DeviceArena,
    sources: &[ArenaSlice],
    destination: ArenaSlice,
) -> Result<(), ResidentLaunchError> {
    let required_words = sources
        .len()
        .checked_mul(4)
        .ok_or(ResidentLaunchError::Binding("claimed-sum width overflow"))?;
    if destination.len_words() != required_words {
        return Err(ResidentLaunchError::Binding(
            "claimed-sum transcript width mismatch",
        ));
    }
    for (index, source) in sources.iter().copied().enumerate() {
        if source.len_words() < 4 {
            return Err(ResidentLaunchError::Binding(
                "claimed-sum source is too small",
            ));
        }
        let offset = index
            .checked_mul(4)
            .ok_or(ResidentLaunchError::Binding("claimed-sum offset overflow"))?;
        // SAFETY: exact aggregate width and each four-word source were checked.
        unsafe {
            arena
                .context()
                .memcpy_d2d_async(
                    destination.as_u32_ptr().add(offset).cast(),
                    source.as_void_ptr().cast_const(),
                    4 * core::mem::size_of::<u32>(),
                )
                .map_err(ResidentLaunchError::Cuda)?;
        }
    }
    Ok(())
}

fn interaction_outputs_in_cairo_order(
    workspace: &GraphWorkspace,
    relation: &PreparedRelationGraph<'_>,
) -> Result<Vec<ArenaSlice>, ResidentRuntimeError> {
    let execution = &workspace.plan().relation().execution;
    let mut outputs = relation.outputs().collect::<Vec<_>>();
    outputs.sort_by_key(|output| {
        let batch = execution.batches[output.batch_index];
        let component = crate::schedule_table::CAIRO_COMMITMENT_COMPONENT_ORDER
            .iter()
            .position(|candidate| *candidate == batch.component)
            .unwrap_or(usize::MAX);
        let (part, instance) = match batch.trace_part {
            RelationTracePart::Component => (0u8, output.instance_index),
            RelationTracePart::EachMemoryBig => (1, output.instance_index),
            RelationTracePart::MemorySmall => (2, output.instance_index),
        };
        (component, part, instance)
    });
    for output in &outputs {
        let batch = execution.batches[output.batch_index];
        if !crate::schedule_table::CAIRO_COMMITMENT_COMPONENT_ORDER.contains(&batch.component) {
            return Err(ResidentRuntimeError::UnknownTranscriptRelationComponent(
                batch.component,
            ));
        }
    }
    Ok(outputs
        .into_iter()
        .map(|output| output.claimed_sum)
        .collect())
}

fn commitment_groups(
    workspace: &GraphWorkspace,
    planned: &PlannedCommitment,
) -> Result<Vec<CommitCoefficientGroup>, ResidentRuntimeError> {
    planned
        .grouped_column_sources
        .iter()
        .zip(&planned.grouped_column_log_sizes)
        .map(|(sources, logs)| {
            let columns = sources
                .iter()
                .zip(logs)
                .map(|(&source, &log_size)| {
                    let binding = match source {
                        CommitmentColumnSource::Trace {
                            component,
                            part,
                            purpose,
                            ordinal,
                        } => workspace
                            .plan()
                            .find(Some(component), Some(part), purpose, ordinal)
                            .map(|(_, binding)| binding),
                        CommitmentColumnSource::Composition { ordinal } => workspace
                            .plan()
                            .find(None, None, BufferPurpose::CompositionCoefficients, ordinal)
                            .map(|(_, binding)| binding),
                    }
                    .ok_or(ResidentRuntimeError::MissingCommitmentSource {
                        id: planned.id,
                        source,
                    })?;
                    let coefficients = workspace.arena().bind(binding.physical)?;
                    require_resident_source(workspace, coefficients)?;
                    Ok(CommitCoefficientColumn {
                        coefficients,
                        log_size,
                    })
                })
                .collect::<Result<Vec<_>, ResidentRuntimeError>>()?;
            Ok(CommitCoefficientGroup { columns })
        })
        .collect()
}

fn arena_relation_sources(
    workspace: &GraphWorkspace,
) -> Result<Vec<RelationInstanceSources>, ResidentRuntimeError> {
    let relation = workspace.plan().relation();
    let mut ordered = Vec::with_capacity(relation.requirements.instances.len());
    for requirement in &relation.requirements.instances {
        let batch = relation.execution.batches[requirement.batch_index];
        let kernel_batch = &relation.execution.kernel_program().batches[requirement.batch_index];
        let part = match batch.trace_part {
            RelationTracePart::Component => {
                stwo_cairo_prover::witness::proof_shape::TracePartId::Main
            }
            RelationTracePart::EachMemoryBig => {
                stwo_cairo_prover::witness::proof_shape::TracePartId::MemoryBig(
                    u32::try_from(requirement.instance_index).map_err(|_| {
                        ResidentRuntimeError::MissingRelationSource {
                            batch,
                            instance_index: requirement.instance_index,
                            ordinal: u32::MAX,
                        }
                    })?,
                )
            }
            RelationTracePart::MemorySmall => {
                stwo_cairo_prover::witness::proof_shape::TracePartId::MemorySmall
            }
        };
        let (purpose, count) = match kernel_batch.source_layout {
            RelationSourceLayout::LookupWords { .. } => (BufferPurpose::LookupInputs, 1),
            RelationSourceLayout::MemoryAddress { chunks } => (
                BufferPurpose::BaseTrace,
                chunks
                    .checked_mul(2)
                    .ok_or(ResidentRuntimeError::MissingRelationSource {
                        batch,
                        instance_index: requirement.instance_index,
                        ordinal: u32::MAX,
                    })?,
            ),
            RelationSourceLayout::MemoryBig { value_words }
            | RelationSourceLayout::MemorySmall { value_words } => (
                BufferPurpose::BaseTrace,
                value_words
                    .checked_add(1)
                    .ok_or(ResidentRuntimeError::MissingRelationSource {
                        batch,
                        instance_index: requirement.instance_index,
                        ordinal: u32::MAX,
                    })?,
            ),
            RelationSourceLayout::BitwiseXor12 {
                multiplicity_columns,
            } => (BufferPurpose::BaseTrace, multiplicity_columns),
        };
        let columns = (0..count)
            .map(|ordinal| {
                let (_, binding) = workspace
                    .plan()
                    .find(Some(batch.component), Some(part), purpose, ordinal)
                    .ok_or(ResidentRuntimeError::MissingRelationSource {
                        batch,
                        instance_index: requirement.instance_index,
                        ordinal,
                    })?;
                let source = workspace.arena().bind(binding.physical)?;
                require_resident_source(workspace, source)?;
                Ok(source)
            })
            .collect::<Result<Vec<_>, ResidentRuntimeError>>()?;
        ordered.push(RelationInstanceSources { columns });
    }
    Ok(ordered)
}

fn transcript_bindings(
    workspace: &GraphWorkspace,
) -> Result<
    (
        Vec<(TranscriptInputId, ArenaSlice)>,
        Vec<(TranscriptOutputId, ArenaSlice)>,
    ),
    ResidentRuntimeError,
> {
    let planned = workspace.plan().transcript();
    let inputs = planned
        .inputs
        .iter()
        .map(|&(id, binding)| Ok((id, workspace.arena().bind(binding.physical)?)))
        .collect::<Result<Vec<_>, ResidentRuntimeError>>()?;
    let outputs = planned
        .outputs
        .iter()
        .map(|&(id, binding)| Ok((id, workspace.arena().bind(binding.physical)?)))
        .collect::<Result<Vec<_>, ResidentRuntimeError>>()?;
    Ok((inputs, outputs))
}

fn require_resident_source(
    workspace: &GraphWorkspace,
    source: ArenaSlice,
) -> Result<(), ResidentRuntimeError> {
    let arena_start = workspace.arena().base_ptr().as_ptr() as usize;
    let arena_end = workspace
        .plan()
        .total_words()
        .checked_mul(core::mem::size_of::<u32>())
        .and_then(|bytes| arena_start.checked_add(bytes));
    let source_start = source.as_u32_ptr() as usize;
    let source_end = source_start.checked_add(source.len_bytes());
    let (Some(arena_end), Some(source_end)) = (arena_end, source_end) else {
        return Err(ResidentRuntimeError::SourceOutsideArena { slot: source.id() });
    };
    if source_start < arena_start || source_end > arena_end {
        return Err(ResidentRuntimeError::SourceOutsideArena { slot: source.id() });
    }
    Ok(())
}

fn fri_round_segment(round_index: usize) -> Result<GraphSegment, ResidentRuntimeError> {
    let layer = round_index
        .checked_add(1)
        .and_then(|value| u8::try_from(value).ok())
        .ok_or(ResidentRuntimeError::FriRoundIndexTooLarge(round_index))?;
    Ok(GraphSegment::FriLayer(layer))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fri_segments_reserve_zero_for_the_original_tree() {
        assert_eq!(fri_round_segment(0).unwrap(), GraphSegment::FriLayer(1));
        assert_eq!(fri_round_segment(254).unwrap(), GraphSegment::FriLayer(255));
        assert!(matches!(
            fri_round_segment(255),
            Err(ResidentRuntimeError::FriRoundIndexTooLarge(255))
        ));
    }

    #[test]
    fn workspace_identity_includes_arena_not_only_protocol_shape() {
        let first = ResidentWorkspaceIdentity {
            shape_key: ProofShapeKey(7),
            protocol_key: 11,
            arena_base: 13,
        };
        assert_ne!(
            first,
            ResidentWorkspaceIdentity {
                arena_base: 17,
                ..first
            }
        );
    }
}
