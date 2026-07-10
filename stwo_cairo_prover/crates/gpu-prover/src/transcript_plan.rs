//! Pure ordinary-Blake2s transcript schedule for one Cairo proof.
//!
//! This module only describes protocol order and stable bindings. CUDA state,
//! arena allocation, graph capture and proof assembly belong to their runtime
//! layers. Keeping this plan pure lets the host channel remain the executable
//! transcript oracle during the device-channel migration.

use core::ops::Range;

use cairo_air::claims::{CairoClaim, CairoInteractionClaim};
use cairo_air::utils::pack_into_secure_felts;
use cairo_air::verifier::INTERACTION_POW_BITS;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::{SecureField, SECURE_EXTENSION_DEGREE};
use stwo::core::pcs::{PcsConfig, TreeVec};
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleHasher;
use stwo::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;
use stwo::core::ColumnVec;
use stwo_backend_cuda::{
    fri_workspace_requirements, Blake2sTranscriptSchedule, DeviceTranscriptError,
    FriWorkspaceConfig, PreparedFriError, TranscriptBoundaryId, TranscriptInputId,
    TranscriptOperation, TranscriptOutputId, TranscriptStart,
};

pub const CAIRO_BLAKE2S_TRANSCRIPT_SCHEDULE_TAG: &str = "stwo-cairo.blake2s.transcript.schedule.v1";
const MAX_REJECTION_ROUNDS: u32 = 64;
const FRI_ID_BASE: u32 = 0x1_0000;
const FRI_ID_STRIDE: u32 = 4;

/// Values which are produced outside the channel and bound to transcript
/// operations. Numeric mappings are protocol ABI and must never be reordered.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum CairoTranscriptInput {
    ChannelSalt,
    PcsConfig,
    PreprocessedRoot,
    ClaimComponentCount,
    ClaimEnableBits,
    ClaimLogSizes,
    ClaimProgramLength,
    ClaimPublicData,
    ClaimOutputRoot,
    ClaimProgramRoot,
    BaseRoot,
    InteractionPowNonce,
    InteractionClaim,
    InteractionRoot,
    CompositionRoot,
    OodsSampledValues,
    FriLayerRoot(u32),
    FriLastLayerPolynomial,
    QueryPowNonce,
}

impl CairoTranscriptInput {
    pub fn id(self) -> Result<TranscriptInputId, TranscriptPlanError> {
        let id = match self {
            Self::ChannelSalt => 1,
            Self::PcsConfig => 2,
            Self::PreprocessedRoot => 3,
            Self::ClaimComponentCount => 10,
            Self::ClaimEnableBits => 11,
            Self::ClaimLogSizes => 12,
            Self::ClaimProgramLength => 13,
            Self::ClaimPublicData => 14,
            Self::ClaimOutputRoot => 15,
            Self::ClaimProgramRoot => 16,
            Self::BaseRoot => 20,
            Self::InteractionPowNonce => 21,
            Self::InteractionClaim => 22,
            Self::InteractionRoot => 23,
            Self::CompositionRoot => 24,
            Self::OodsSampledValues => 25,
            Self::FriLayerRoot(layer) => fri_id(layer, 0)?,
            Self::FriLastLayerPolynomial => 30,
            Self::QueryPowNonce => 31,
        };
        Ok(TranscriptInputId(id))
    }
}

/// Challenge and query destinations written by the device channel.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum CairoTranscriptOutput {
    CommonLookupElements,
    CompositionRandomCoefficient,
    OodsPointParameter,
    QuotientRandomCoefficient,
    FriFoldingChallenge(u32),
    QueryPositions,
}

impl CairoTranscriptOutput {
    pub fn id(self) -> Result<TranscriptOutputId, TranscriptPlanError> {
        let id = match self {
            Self::CommonLookupElements => 1,
            Self::CompositionRandomCoefficient => 2,
            Self::OodsPointParameter => 3,
            Self::QuotientRandomCoefficient => 4,
            Self::FriFoldingChallenge(layer) => fri_id(layer, 1)?,
            Self::QueryPositions => 5,
        };
        Ok(TranscriptOutputId(id))
    }
}

/// Stable semantic name for each exact host-channel call.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CairoTranscriptBoundary {
    ChannelSalt,
    PcsConfig,
    PreprocessedRoot,
    ClaimComponentCount,
    ClaimEnableBits,
    ClaimLogSizes,
    ClaimProgramLength,
    ClaimPublicData,
    ClaimOutputRoot,
    ClaimProgramRoot,
    BaseRoot,
    InteractionPow,
    CommonLookupElements,
    InteractionClaim,
    InteractionRoot,
    CompositionRandomCoefficient,
    CompositionRoot,
    OodsPoint,
    OodsSampledValues,
    QuotientRandomCoefficient,
    FriLayerRoot(u32),
    FriFoldingChallenge(u32),
    FriLastLayerPolynomial,
    QueryPow,
    QueryPositions,
}

impl CairoTranscriptBoundary {
    pub fn id(self) -> Result<TranscriptBoundaryId, TranscriptPlanError> {
        let id = match self {
            Self::ChannelSalt => 1,
            Self::PcsConfig => 2,
            Self::PreprocessedRoot => 3,
            Self::ClaimComponentCount => 10,
            Self::ClaimEnableBits => 11,
            Self::ClaimLogSizes => 12,
            Self::ClaimProgramLength => 13,
            Self::ClaimPublicData => 14,
            Self::ClaimOutputRoot => 15,
            Self::ClaimProgramRoot => 16,
            Self::BaseRoot => 20,
            Self::InteractionPow => 21,
            Self::CommonLookupElements => 22,
            Self::InteractionClaim => 30,
            Self::InteractionRoot => 31,
            Self::CompositionRandomCoefficient => 32,
            Self::CompositionRoot => 40,
            Self::OodsPoint => 41,
            Self::OodsSampledValues => 50,
            Self::QuotientRandomCoefficient => 51,
            Self::FriLayerRoot(layer) => fri_id(layer, 2)?,
            Self::FriFoldingChallenge(layer) => fri_id(layer, 3)?,
            Self::FriLastLayerPolynomial => 60,
            Self::QueryPow => 61,
            Self::QueryPositions => 62,
        };
        Ok(TranscriptBoundaryId(id))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CairoTranscriptSegment {
    /// Static inputs through the freshly committed base-tree root. The device
    /// transcript state after this segment is the exact interaction-PoW seed.
    BootstrapThroughBase,
    /// Absorb the device-found interaction nonce, then draw z and alpha.
    InteractionPowAndLookup,
    InteractionAndComposition,
    CompositionAndOods,
    OodsAndQuotient,
    FriLayer(u32),
    /// Absorb the device-produced final LinePoly. The resulting state is the
    /// exact query-PoW seed.
    FriLastLayer,
    /// Absorb the device-found query nonce, then draw query positions.
    QueryPowAndPositions,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TranscriptSegmentPlan {
    pub segment: CairoTranscriptSegment,
    pub operation_range: Range<usize>,
    pub starts_after: Option<CairoTranscriptBoundary>,
    pub ends_at: CairoTranscriptBoundary,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TranscriptBoundaryPlan {
    pub semantic: CairoTranscriptBoundary,
    pub operation_index: usize,
    pub segment: CairoTranscriptSegment,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TranscriptInputRequirement {
    pub semantic: CairoTranscriptInput,
    pub min_words: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TranscriptOutputRequirement {
    pub semantic: CairoTranscriptOutput,
    pub min_words: usize,
}

/// Sizes that only become known once witness/relation/OODS planning has
/// completed. `None` is intentionally a typed fail-closed boundary.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DynamicTranscriptShape {
    pub interaction_claim_felts: Option<usize>,
    pub oods_sampled_values_felts: Option<usize>,
}

impl DynamicTranscriptShape {
    /// Derive both dynamic mix lengths from the exact protocol values. This is
    /// the preferred post-OODS constructor and cannot drift from flatten order.
    pub fn from_protocol_values(
        interaction_claim: &CairoInteractionClaim,
        sampled_values: &TreeVec<ColumnVec<Vec<SecureField>>>,
    ) -> Result<Self, TranscriptPlanError> {
        let sampled_values_felts = sampled_values
            .0
            .iter()
            .flatten()
            .try_fold(0usize, |total, column| total.checked_add(column.len()))
            .ok_or(TranscriptPlanError::SizeOverflow)?;
        Ok(Self {
            interaction_claim_felts: Some(interaction_claim.flatten_interaction_claim().len()),
            oods_sampled_values_felts: Some(sampled_values_felts),
        })
    }

    /// Pre-OODS planning may bind the exact interaction claim while leaving
    /// the OODS length as an explicit pending boundary.
    pub fn from_interaction_claim(interaction_claim: &CairoInteractionClaim) -> Self {
        Self {
            interaction_claim_felts: Some(interaction_claim.flatten_interaction_claim().len()),
            oods_sampled_values_felts: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PendingTranscriptBoundary {
    InteractionClaim,
    OodsSampledValues,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClaimMixSegment {
    ComponentEnableBits,
    ComponentLogSizes,
    PublicData,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TranscriptPlanError {
    Pending(PendingTranscriptBoundary),
    EmptyClaimSegment(ClaimMixSegment),
    EmptyInteractionClaim,
    EmptyOodsSampledValues,
    EmptyFriSchedule,
    InvalidQueryCount,
    SizeOverflow,
    Fri(PreparedFriError),
    Device(DeviceTranscriptError),
}

impl core::fmt::Display for TranscriptPlanError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "invalid Cairo Blake2s transcript plan: {self:?}")
    }
}

impl std::error::Error for TranscriptPlanError {}

impl From<PreparedFriError> for TranscriptPlanError {
    fn from(value: PreparedFriError) -> Self {
        Self::Fri(value)
    }
}

impl From<DeviceTranscriptError> for TranscriptPlanError {
    fn from(value: DeviceTranscriptError) -> Self {
        Self::Device(value)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ClaimMixShape {
    enable_felts: u32,
    log_size_felts: u32,
    public_data_felts: u32,
}

impl ClaimMixShape {
    fn from_claim(claim: &CairoClaim) -> Result<Self, TranscriptPlanError> {
        let flat = claim.flatten_claim();
        let (public_data, ..) = claim.public_data.pack_into_u32s();
        Ok(Self {
            enable_felts: packed_felt_count(
                flat.component_enable_bits.len(),
                ClaimMixSegment::ComponentEnableBits,
            )?,
            log_size_felts: packed_felt_count(
                flat.component_log_sizes.len(),
                ClaimMixSegment::ComponentLogSizes,
            )?,
            public_data_felts: packed_felt_count(public_data.len(), ClaimMixSegment::PublicData)?,
        })
    }
}

/// Complete protocol schedule plus graph-safe transcript dependency segments.
#[derive(Clone, Debug)]
pub struct CairoBlake2sTranscriptPlan {
    schedule: Blake2sTranscriptSchedule,
    schedule_key: u64,
    inputs: Vec<TranscriptInputRequirement>,
    outputs: Vec<TranscriptOutputRequirement>,
    boundaries: Vec<TranscriptBoundaryPlan>,
    segments: Vec<TranscriptSegmentPlan>,
}

impl CairoBlake2sTranscriptPlan {
    pub fn schedule(&self) -> &Blake2sTranscriptSchedule {
        &self.schedule
    }

    pub fn schedule_key(&self) -> u64 {
        self.schedule_key
    }

    pub fn inputs(&self) -> &[TranscriptInputRequirement] {
        &self.inputs
    }

    pub fn outputs(&self) -> &[TranscriptOutputRequirement] {
        &self.outputs
    }

    pub fn boundaries(&self) -> &[TranscriptBoundaryPlan] {
        &self.boundaries
    }

    pub fn segments(&self) -> &[TranscriptSegmentPlan] {
        &self.segments
    }
}

/// Derives the exact ordinary-Blake2s operation order used by
/// `prove_cairo_common`, `prove_ex` and the typed PCS proof driver.
pub fn plan_cairo_blake2s_transcript(
    claim: &CairoClaim,
    pcs: PcsConfig,
    lifting_log_size: u32,
    dynamic: DynamicTranscriptShape,
) -> Result<CairoBlake2sTranscriptPlan, TranscriptPlanError> {
    plan_with_interaction_pow_bits(
        ClaimMixShape::from_claim(claim)?,
        pcs,
        lifting_log_size,
        dynamic,
        INTERACTION_POW_BITS,
    )
}

fn plan_with_interaction_pow_bits(
    claim: ClaimMixShape,
    pcs: PcsConfig,
    lifting_log_size: u32,
    dynamic: DynamicTranscriptShape,
    interaction_pow_bits: u32,
) -> Result<CairoBlake2sTranscriptPlan, TranscriptPlanError> {
    let interaction_claim_felts =
        dynamic
            .interaction_claim_felts
            .ok_or(TranscriptPlanError::Pending(
                PendingTranscriptBoundary::InteractionClaim,
            ))?;
    let sampled_values_felts =
        dynamic
            .oods_sampled_values_felts
            .ok_or(TranscriptPlanError::Pending(
                PendingTranscriptBoundary::OodsSampledValues,
            ))?;
    let interaction_claim_felts = nonzero_u32(
        interaction_claim_felts,
        TranscriptPlanError::EmptyInteractionClaim,
    )?;
    let sampled_values_felts = nonzero_u32(
        sampled_values_felts,
        TranscriptPlanError::EmptyOodsSampledValues,
    )?;
    let n_queries =
        u32::try_from(pcs.fri_config.n_queries).map_err(|_| TranscriptPlanError::SizeOverflow)?;
    if n_queries == 0 {
        return Err(TranscriptPlanError::InvalidQueryCount);
    }
    let twiddle_log_size = lifting_log_size
        .checked_sub(1)
        .ok_or(TranscriptPlanError::SizeOverflow)?;
    let fri = fri_workspace_requirements(FriWorkspaceConfig {
        fri: pcs.fri_config,
        circle_log_size: lifting_log_size,
        twiddle_log_size,
    })?;
    if fri.trees.is_empty() {
        return Err(TranscriptPlanError::EmptyFriSchedule);
    }
    let last_layer_felts = 1usize
        .checked_shl(pcs.fri_config.log_last_layer_degree_bound)
        .ok_or(TranscriptPlanError::SizeOverflow)?;
    let last_layer_felts =
        u32::try_from(last_layer_felts).map_err(|_| TranscriptPlanError::SizeOverflow)?;

    let mut builder = PlanBuilder::default();
    builder.begin(CairoTranscriptSegment::BootstrapThroughBase, None);
    builder.mix_felts(
        CairoTranscriptBoundary::ChannelSalt,
        CairoTranscriptInput::ChannelSalt,
        1,
    )?;
    builder.mix_felts(
        CairoTranscriptBoundary::PcsConfig,
        CairoTranscriptInput::PcsConfig,
        2,
    )?;
    builder.absorb_root(
        CairoTranscriptBoundary::PreprocessedRoot,
        CairoTranscriptInput::PreprocessedRoot,
    )?;
    builder.mix_felts(
        CairoTranscriptBoundary::ClaimComponentCount,
        CairoTranscriptInput::ClaimComponentCount,
        1,
    )?;
    builder.mix_felts(
        CairoTranscriptBoundary::ClaimEnableBits,
        CairoTranscriptInput::ClaimEnableBits,
        claim.enable_felts,
    )?;
    builder.mix_felts(
        CairoTranscriptBoundary::ClaimLogSizes,
        CairoTranscriptInput::ClaimLogSizes,
        claim.log_size_felts,
    )?;
    builder.mix_felts(
        CairoTranscriptBoundary::ClaimProgramLength,
        CairoTranscriptInput::ClaimProgramLength,
        1,
    )?;
    builder.mix_felts(
        CairoTranscriptBoundary::ClaimPublicData,
        CairoTranscriptInput::ClaimPublicData,
        claim.public_data_felts,
    )?;
    builder.absorb_root(
        CairoTranscriptBoundary::ClaimOutputRoot,
        CairoTranscriptInput::ClaimOutputRoot,
    )?;
    builder.absorb_root(
        CairoTranscriptBoundary::ClaimProgramRoot,
        CairoTranscriptInput::ClaimProgramRoot,
    )?;
    builder.absorb_root(
        CairoTranscriptBoundary::BaseRoot,
        CairoTranscriptInput::BaseRoot,
    )?;
    builder.end(CairoTranscriptBoundary::BaseRoot);

    builder.begin(
        CairoTranscriptSegment::InteractionPowAndLookup,
        Some(CairoTranscriptBoundary::BaseRoot),
    );
    builder.absorb_pow(
        CairoTranscriptBoundary::InteractionPow,
        CairoTranscriptInput::InteractionPowNonce,
        interaction_pow_bits,
    )?;
    builder.draw_secure_felts(
        CairoTranscriptBoundary::CommonLookupElements,
        CairoTranscriptOutput::CommonLookupElements,
        2,
    )?;
    builder.end(CairoTranscriptBoundary::CommonLookupElements);

    builder.begin(
        CairoTranscriptSegment::InteractionAndComposition,
        Some(CairoTranscriptBoundary::CommonLookupElements),
    );
    builder.mix_felts(
        CairoTranscriptBoundary::InteractionClaim,
        CairoTranscriptInput::InteractionClaim,
        interaction_claim_felts,
    )?;
    builder.absorb_root(
        CairoTranscriptBoundary::InteractionRoot,
        CairoTranscriptInput::InteractionRoot,
    )?;
    builder.draw_secure_felt(
        CairoTranscriptBoundary::CompositionRandomCoefficient,
        CairoTranscriptOutput::CompositionRandomCoefficient,
    )?;
    builder.end(CairoTranscriptBoundary::CompositionRandomCoefficient);

    builder.begin(
        CairoTranscriptSegment::CompositionAndOods,
        Some(CairoTranscriptBoundary::CompositionRandomCoefficient),
    );
    builder.absorb_root(
        CairoTranscriptBoundary::CompositionRoot,
        CairoTranscriptInput::CompositionRoot,
    )?;
    builder.draw_secure_felt(
        CairoTranscriptBoundary::OodsPoint,
        CairoTranscriptOutput::OodsPointParameter,
    )?;
    builder.end(CairoTranscriptBoundary::OodsPoint);

    builder.begin(
        CairoTranscriptSegment::OodsAndQuotient,
        Some(CairoTranscriptBoundary::OodsPoint),
    );
    builder.mix_felts(
        CairoTranscriptBoundary::OodsSampledValues,
        CairoTranscriptInput::OodsSampledValues,
        sampled_values_felts,
    )?;
    builder.draw_secure_felt(
        CairoTranscriptBoundary::QuotientRandomCoefficient,
        CairoTranscriptOutput::QuotientRandomCoefficient,
    )?;
    builder.end(CairoTranscriptBoundary::QuotientRandomCoefficient);

    let mut previous = CairoTranscriptBoundary::QuotientRandomCoefficient;
    for layer in 0..u32::try_from(fri.trees.len()).map_err(|_| TranscriptPlanError::SizeOverflow)? {
        builder.begin(CairoTranscriptSegment::FriLayer(layer), Some(previous));
        builder.absorb_root(
            CairoTranscriptBoundary::FriLayerRoot(layer),
            CairoTranscriptInput::FriLayerRoot(layer),
        )?;
        let challenge = CairoTranscriptBoundary::FriFoldingChallenge(layer);
        builder.draw_secure_felt(challenge, CairoTranscriptOutput::FriFoldingChallenge(layer))?;
        builder.end(challenge);
        previous = challenge;
    }

    builder.begin(CairoTranscriptSegment::FriLastLayer, Some(previous));
    builder.mix_felts(
        CairoTranscriptBoundary::FriLastLayerPolynomial,
        CairoTranscriptInput::FriLastLayerPolynomial,
        last_layer_felts,
    )?;
    builder.end(CairoTranscriptBoundary::FriLastLayerPolynomial);

    builder.begin(
        CairoTranscriptSegment::QueryPowAndPositions,
        Some(CairoTranscriptBoundary::FriLastLayerPolynomial),
    );
    builder.absorb_pow(
        CairoTranscriptBoundary::QueryPow,
        CairoTranscriptInput::QueryPowNonce,
        pcs.pow_bits,
    )?;
    builder.draw_queries(
        CairoTranscriptBoundary::QueryPositions,
        CairoTranscriptOutput::QueryPositions,
        lifting_log_size,
        n_queries,
    )?;
    builder.end(CairoTranscriptBoundary::QueryPositions);
    builder.finish()
}

#[derive(Default)]
struct PlanBuilder {
    operations: Vec<TranscriptOperation>,
    inputs: Vec<TranscriptInputRequirement>,
    outputs: Vec<TranscriptOutputRequirement>,
    boundaries: Vec<TranscriptBoundaryPlan>,
    segments: Vec<TranscriptSegmentPlan>,
    current: Option<(
        CairoTranscriptSegment,
        usize,
        Option<CairoTranscriptBoundary>,
    )>,
}

impl PlanBuilder {
    fn begin(
        &mut self,
        segment: CairoTranscriptSegment,
        starts_after: Option<CairoTranscriptBoundary>,
    ) {
        assert!(self.current.is_none());
        self.current = Some((segment, self.operations.len(), starts_after));
    }

    fn end(&mut self, ends_at: CairoTranscriptBoundary) {
        let (segment, start, starts_after) = self.current.take().expect("segment is open");
        self.segments.push(TranscriptSegmentPlan {
            segment,
            operation_range: start..self.operations.len(),
            starts_after,
            ends_at,
        });
    }

    fn push(&mut self, semantic: CairoTranscriptBoundary, operation: TranscriptOperation) {
        let segment = self.current.expect("operation belongs to a segment").0;
        let operation_index = self.operations.len();
        self.operations.push(operation);
        self.boundaries.push(TranscriptBoundaryPlan {
            semantic,
            operation_index,
            segment,
        });
    }

    fn require_input(&mut self, semantic: CairoTranscriptInput, min_words: usize) {
        self.inputs.push(TranscriptInputRequirement {
            semantic,
            min_words,
        });
    }

    fn require_output(&mut self, semantic: CairoTranscriptOutput, min_words: usize) {
        self.outputs.push(TranscriptOutputRequirement {
            semantic,
            min_words,
        });
    }

    fn mix_felts(
        &mut self,
        boundary: CairoTranscriptBoundary,
        input: CairoTranscriptInput,
        n_felts: u32,
    ) -> Result<(), TranscriptPlanError> {
        self.require_input(input, words_for_felts(n_felts)?);
        self.push(
            boundary,
            TranscriptOperation::MixFelts {
                boundary: boundary.id()?,
                source: input.id()?,
                n_felts,
            },
        );
        Ok(())
    }

    fn absorb_root(
        &mut self,
        boundary: CairoTranscriptBoundary,
        input: CairoTranscriptInput,
    ) -> Result<(), TranscriptPlanError> {
        self.require_input(input, 8);
        self.push(
            boundary,
            TranscriptOperation::AbsorbRoot {
                boundary: boundary.id()?,
                source: input.id()?,
            },
        );
        Ok(())
    }

    fn absorb_pow(
        &mut self,
        boundary: CairoTranscriptBoundary,
        input: CairoTranscriptInput,
        pow_bits: u32,
    ) -> Result<(), TranscriptPlanError> {
        self.require_input(input, 2);
        self.push(
            boundary,
            TranscriptOperation::AbsorbPowNonce {
                boundary: boundary.id()?,
                source: input.id()?,
                pow_bits,
            },
        );
        Ok(())
    }

    fn draw_secure_felt(
        &mut self,
        boundary: CairoTranscriptBoundary,
        output: CairoTranscriptOutput,
    ) -> Result<(), TranscriptPlanError> {
        self.require_output(output, SECURE_EXTENSION_DEGREE);
        self.push(
            boundary,
            TranscriptOperation::DrawSecureFelt {
                boundary: boundary.id()?,
                output: output.id()?,
            },
        );
        Ok(())
    }

    fn draw_secure_felts(
        &mut self,
        boundary: CairoTranscriptBoundary,
        output: CairoTranscriptOutput,
        n_felts: u32,
    ) -> Result<(), TranscriptPlanError> {
        self.require_output(output, words_for_felts(n_felts)?);
        self.push(
            boundary,
            TranscriptOperation::DrawSecureFelts {
                boundary: boundary.id()?,
                output: output.id()?,
                n_felts,
            },
        );
        Ok(())
    }

    fn draw_queries(
        &mut self,
        boundary: CairoTranscriptBoundary,
        output: CairoTranscriptOutput,
        log_domain_size: u32,
        n_queries: u32,
    ) -> Result<(), TranscriptPlanError> {
        self.require_output(
            output,
            usize::try_from(n_queries).map_err(|_| TranscriptPlanError::SizeOverflow)?,
        );
        self.push(
            boundary,
            TranscriptOperation::DrawQueries {
                boundary: boundary.id()?,
                output: output.id()?,
                log_domain_size,
                n_queries,
            },
        );
        Ok(())
    }

    fn finish(self) -> Result<CairoBlake2sTranscriptPlan, TranscriptPlanError> {
        assert!(self.current.is_none());
        let schedule = Blake2sTranscriptSchedule::new(
            TranscriptStart::Default,
            self.operations,
            MAX_REJECTION_ROUNDS,
        )?;
        let schedule_key = cairo_schedule_key(&schedule, &self.segments);
        Ok(CairoBlake2sTranscriptPlan {
            schedule,
            schedule_key,
            inputs: self.inputs,
            outputs: self.outputs,
            boundaries: self.boundaries,
            segments: self.segments,
        })
    }
}

/// Exact host/reference words for the salt, PCS config and Cairo claim inputs.
/// Commitment roots, PoW nonces and later proof values are produced by their
/// owning graph segments and therefore are not synthesized here.
pub fn encode_static_transcript_inputs(
    channel_salt: u32,
    pcs: PcsConfig,
    claim: &CairoClaim,
) -> Result<Vec<(CairoTranscriptInput, Vec<u32>)>, TranscriptPlanError> {
    let flat = claim.flatten_claim();
    let (public_data, output, program) = claim.public_data.pack_into_u32s();
    // Validate all dynamic claim segment sizes through the same production
    // planner checks before returning any binding material.
    let _ = ClaimMixShape::from_claim(claim)?;

    let component_count = u32::try_from(flat.component_enable_bits.len())
        .map_err(|_| TranscriptPlanError::SizeOverflow)?;
    let program_length = u32::try_from(claim.public_data.public_memory.program.len())
        .map_err(|_| TranscriptPlanError::SizeOverflow)?;
    let salt: SecureField = channel_salt.into();
    let config = [
        SecureField::from_u32_unchecked(
            pcs.pow_bits,
            pcs.fri_config.log_blowup_factor,
            u32::try_from(pcs.fri_config.n_queries)
                .map_err(|_| TranscriptPlanError::SizeOverflow)?,
            pcs.fri_config.log_last_layer_degree_bound,
        ),
        SecureField::from_u32_unchecked(
            pcs.fri_config.fold_step,
            pcs.lifting_log_size.unwrap_or(0),
            0,
            0,
        ),
    ];
    Ok(vec![
        (CairoTranscriptInput::ChannelSalt, felts_to_words(&[salt])),
        (CairoTranscriptInput::PcsConfig, felts_to_words(&config)),
        (
            CairoTranscriptInput::ClaimComponentCount,
            felts_to_words(&pack_into_secure_felts([component_count].into_iter())),
        ),
        (
            CairoTranscriptInput::ClaimEnableBits,
            felts_to_words(&pack_into_secure_felts(
                flat.component_enable_bits
                    .iter()
                    .map(|&enabled| u32::from(enabled)),
            )),
        ),
        (
            CairoTranscriptInput::ClaimLogSizes,
            felts_to_words(&pack_into_secure_felts(
                flat.component_log_sizes.iter().copied(),
            )),
        ),
        (
            CairoTranscriptInput::ClaimProgramLength,
            felts_to_words(&pack_into_secure_felts([program_length].into_iter())),
        ),
        (
            CairoTranscriptInput::ClaimPublicData,
            felts_to_words(&pack_into_secure_felts(public_data.into_iter())),
        ),
        (
            CairoTranscriptInput::ClaimOutputRoot,
            merkle_leaf_root_words(&output),
        ),
        (
            CairoTranscriptInput::ClaimProgramRoot,
            merkle_leaf_root_words(&program),
        ),
    ])
}

/// Exact dynamic interaction-claim serialization used by
/// `CairoInteractionClaim::mix_into`.
pub fn encode_interaction_claim_input(
    claim: &CairoInteractionClaim,
) -> (CairoTranscriptInput, Vec<u32>) {
    (
        CairoTranscriptInput::InteractionClaim,
        felts_to_words(&claim.flatten_interaction_claim()),
    )
}

fn packed_felt_count(values: usize, segment: ClaimMixSegment) -> Result<u32, TranscriptPlanError> {
    if values == 0 {
        return Err(TranscriptPlanError::EmptyClaimSegment(segment));
    }
    u32::try_from(values.div_ceil(SECURE_EXTENSION_DEGREE))
        .map_err(|_| TranscriptPlanError::SizeOverflow)
}

fn nonzero_u32(value: usize, error: TranscriptPlanError) -> Result<u32, TranscriptPlanError> {
    if value == 0 {
        return Err(error);
    }
    u32::try_from(value).map_err(|_| TranscriptPlanError::SizeOverflow)
}

fn words_for_felts(n_felts: u32) -> Result<usize, TranscriptPlanError> {
    usize::try_from(n_felts)
        .ok()
        .and_then(|n| n.checked_mul(SECURE_EXTENSION_DEGREE))
        .ok_or(TranscriptPlanError::SizeOverflow)
}

fn fri_id(layer: u32, offset: u32) -> Result<u32, TranscriptPlanError> {
    layer
        .checked_mul(FRI_ID_STRIDE)
        .and_then(|value| FRI_ID_BASE.checked_add(value))
        .and_then(|value| value.checked_add(offset))
        .ok_or(TranscriptPlanError::SizeOverflow)
}

fn felts_to_words(felts: &[SecureField]) -> Vec<u32> {
    felts
        .iter()
        .flat_map(|felt| felt.to_m31_array())
        .map(|felt| felt.0)
        .collect()
}

fn merkle_leaf_root_words(values: &[u32]) -> Vec<u32> {
    let mut hasher = Blake2sMerkleHasher::default();
    hasher.update_leaf(
        &values
            .iter()
            .map(|&value| BaseField::from_u32_unchecked(value))
            .collect::<Vec<_>>(),
    );
    hasher
        .finalize()
        .0
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn cairo_schedule_key(
    schedule: &Blake2sTranscriptSchedule,
    segments: &[TranscriptSegmentPlan],
) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    let mut feed = |bytes: &[u8]| {
        for &byte in bytes {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    };
    feed(CAIRO_BLAKE2S_TRANSCRIPT_SCHEDULE_TAG.as_bytes());
    feed(&schedule.protocol_key().to_le_bytes());
    for segment in segments {
        let (tag, index) = match segment.segment {
            CairoTranscriptSegment::BootstrapThroughBase => (0u32, 0),
            CairoTranscriptSegment::InteractionPowAndLookup => (1, 0),
            CairoTranscriptSegment::InteractionAndComposition => (2, 0),
            CairoTranscriptSegment::CompositionAndOods => (3, 0),
            CairoTranscriptSegment::OodsAndQuotient => (4, 0),
            CairoTranscriptSegment::FriLayer(index) => (5, index),
            CairoTranscriptSegment::FriLastLayer => (6, 0),
            CairoTranscriptSegment::QueryPowAndPositions => (7, 0),
        };
        feed(&tag.to_le_bytes());
        feed(&index.to_le_bytes());
        feed(&(segment.operation_range.start as u64).to_le_bytes());
        feed(&(segment.operation_range.end as u64).to_le_bytes());
    }
    hash
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use stwo::core::channel::{Blake2sChannel, Channel, MerkleChannel};
    use stwo::core::fri::FriConfig;
    use stwo::core::vcs::blake2_hash::Blake2sHash;
    use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
    use stwo_backend_cuda::replay_blake2s_reference;

    use super::*;

    fn test_pcs() -> PcsConfig {
        PcsConfig {
            pow_bits: 0,
            fri_config: FriConfig::new(2, 1, 13, 2),
            lifting_log_size: Some(10),
        }
    }

    fn test_plan() -> CairoBlake2sTranscriptPlan {
        plan_with_interaction_pow_bits(
            ClaimMixShape {
                enable_felts: 2,
                log_size_felts: 2,
                public_data_felts: 3,
            },
            test_pcs(),
            10,
            DynamicTranscriptShape {
                interaction_claim_felts: Some(5),
                oods_sampled_values_felts: Some(7),
            },
            0,
        )
        .unwrap()
    }

    fn sample_inputs(plan: &CairoBlake2sTranscriptPlan) -> BTreeMap<TranscriptInputId, Vec<u32>> {
        plan.inputs()
            .iter()
            .map(|requirement| {
                let id = requirement.semantic.id().unwrap();
                let words = match requirement.semantic {
                    CairoTranscriptInput::PreprocessedRoot
                    | CairoTranscriptInput::ClaimOutputRoot
                    | CairoTranscriptInput::ClaimProgramRoot
                    | CairoTranscriptInput::BaseRoot
                    | CairoTranscriptInput::InteractionRoot
                    | CairoTranscriptInput::CompositionRoot
                    | CairoTranscriptInput::FriLayerRoot(_) => (0..requirement.min_words)
                        .map(|word| id.0.wrapping_mul(0x10203).wrapping_add(word as u32))
                        .collect(),
                    CairoTranscriptInput::InteractionPowNonce
                    | CairoTranscriptInput::QueryPowNonce => vec![0x1234_5678, 0x9abc_def0],
                    _ => (0..requirement.min_words)
                        .map(|word| (id.0 + word as u32 + 1) % (1 << 20))
                        .collect(),
                };
                (id, words)
            })
            .collect()
    }

    fn snapshot_inputs(
        plan: &CairoBlake2sTranscriptPlan,
        inputs: &BTreeMap<TranscriptInputId, Vec<u32>>,
    ) -> Vec<u32> {
        let mut snapshot = Vec::new();
        for operation in plan.schedule().operations() {
            let source = match *operation {
                TranscriptOperation::MixFelts { source, .. }
                | TranscriptOperation::MixU32s { source, .. }
                | TranscriptOperation::MixU64 { source, .. }
                | TranscriptOperation::AbsorbRoot { source, .. }
                | TranscriptOperation::AbsorbPowNonce { source, .. } => Some(source),
                _ => None,
            };
            if let Some(source) = source {
                snapshot.extend_from_slice(&inputs[&source]);
            }
        }
        snapshot
    }

    fn felts(words: &[u32]) -> Vec<SecureField> {
        words
            .chunks_exact(4)
            .map(|words| {
                SecureField::from_m31_array([
                    BaseField::from_u32_unchecked(words[0]),
                    BaseField::from_u32_unchecked(words[1]),
                    BaseField::from_u32_unchecked(words[2]),
                    BaseField::from_u32_unchecked(words[3]),
                ])
            })
            .collect()
    }

    fn root(words: &[u32]) -> Blake2sHash {
        let mut bytes = [0u8; 32];
        for (bytes, word) in bytes.chunks_exact_mut(4).zip(words) {
            bytes.copy_from_slice(&word.to_le_bytes());
        }
        Blake2sHash(bytes)
    }

    fn direct_host_replay(
        plan: &CairoBlake2sTranscriptPlan,
        inputs: &BTreeMap<TranscriptInputId, Vec<u32>>,
    ) -> (Vec<Blake2sHash>, Vec<u32>) {
        let mut channel = Blake2sChannel::default();
        let mut segment_digests = Vec::new();
        let mut outputs = Vec::new();
        for (index, operation) in plan.schedule().operations().iter().enumerate() {
            match *operation {
                TranscriptOperation::MixFelts { source, .. } => {
                    channel.mix_felts(&felts(&inputs[&source]));
                }
                TranscriptOperation::AbsorbRoot { source, .. } => {
                    Blake2sMerkleChannel::mix_root(&mut channel, root(&inputs[&source]));
                }
                TranscriptOperation::AbsorbPowNonce { source, .. } => {
                    let words = &inputs[&source];
                    channel.mix_u64(u64::from(words[0]) | (u64::from(words[1]) << 32));
                }
                TranscriptOperation::DrawSecureFelt { .. } => {
                    outputs.extend(felts_to_words(&[channel.draw_secure_felt()]));
                }
                TranscriptOperation::DrawSecureFelts { n_felts, .. } => {
                    outputs.extend(felts_to_words(&channel.draw_secure_felts(n_felts as usize)));
                }
                TranscriptOperation::DrawQueries {
                    log_domain_size,
                    n_queries,
                    ..
                } => {
                    let mask = (1u32 << log_domain_size) - 1;
                    let output_words = plan
                        .outputs()
                        .iter()
                        .map(|output| output.min_words)
                        .sum::<usize>();
                    while outputs.len() < output_words {
                        for word in channel.draw_u32s() {
                            outputs.push(word & mask);
                            if outputs.len() == output_words {
                                break;
                            }
                        }
                    }
                    assert_eq!(n_queries as usize, plan.outputs().last().unwrap().min_words);
                }
                TranscriptOperation::MixU32s { .. }
                | TranscriptOperation::MixU64 { .. }
                | TranscriptOperation::DrawU32s { .. } => unreachable!(),
            }
            if plan
                .segments()
                .iter()
                .any(|segment| segment.operation_range.end == index + 1)
            {
                segment_digests.push(channel.digest());
            }
        }
        (segment_digests, outputs)
    }

    #[test]
    fn exact_operation_order_and_segment_boundaries_are_pinned() {
        let plan = test_plan();
        let semantic = plan
            .boundaries()
            .iter()
            .map(|boundary| boundary.semantic)
            .collect::<Vec<_>>();
        assert_eq!(
            &semantic[..13],
            &[
                CairoTranscriptBoundary::ChannelSalt,
                CairoTranscriptBoundary::PcsConfig,
                CairoTranscriptBoundary::PreprocessedRoot,
                CairoTranscriptBoundary::ClaimComponentCount,
                CairoTranscriptBoundary::ClaimEnableBits,
                CairoTranscriptBoundary::ClaimLogSizes,
                CairoTranscriptBoundary::ClaimProgramLength,
                CairoTranscriptBoundary::ClaimPublicData,
                CairoTranscriptBoundary::ClaimOutputRoot,
                CairoTranscriptBoundary::ClaimProgramRoot,
                CairoTranscriptBoundary::BaseRoot,
                CairoTranscriptBoundary::InteractionPow,
                CairoTranscriptBoundary::CommonLookupElements,
            ]
        );
        assert_eq!(plan.segments().first().unwrap().operation_range, 0..11);
        assert_eq!(plan.segments()[1].operation_range, 11..13);
        assert_eq!(
            plan.segments().last().unwrap().ends_at,
            CairoTranscriptBoundary::QueryPositions
        );
        assert!(matches!(
            plan.schedule().operations()[12],
            TranscriptOperation::DrawSecureFelts { n_felts: 2, .. }
        ));
        assert!(matches!(
            plan.schedule().operations().last().unwrap(),
            TranscriptOperation::DrawQueries {
                log_domain_size: 10,
                n_queries: 13,
                ..
            }
        ));
    }

    #[test]
    fn host_reference_replay_matches_every_completed_segment() {
        let plan = test_plan();
        let inputs = sample_inputs(&plan);
        let reference =
            replay_blake2s_reference(&plan.schedule, &snapshot_inputs(&plan, &inputs)).unwrap();
        let (host_digests, host_outputs) = direct_host_replay(&plan, &inputs);
        assert_eq!(reference.output_words, host_outputs);
        assert_eq!(host_digests.len(), plan.segments().len());
        for (host, segment) in host_digests.iter().zip(plan.segments()) {
            assert_eq!(
                *host,
                reference.boundaries[segment.operation_range.end - 1].digest,
                "segment {:?}",
                segment.segment
            );
        }
    }

    #[test]
    fn pending_dynamic_boundaries_fail_closed() {
        let claim = ClaimMixShape {
            enable_felts: 1,
            log_size_felts: 1,
            public_data_felts: 1,
        };
        assert!(matches!(
            plan_with_interaction_pow_bits(
                claim,
                test_pcs(),
                10,
                DynamicTranscriptShape::default(),
                0
            ),
            Err(TranscriptPlanError::Pending(
                PendingTranscriptBoundary::InteractionClaim
            ))
        ));
        assert!(matches!(
            plan_with_interaction_pow_bits(
                claim,
                test_pcs(),
                10,
                DynamicTranscriptShape {
                    interaction_claim_felts: Some(1),
                    oods_sampled_values_felts: None,
                },
                0
            ),
            Err(TranscriptPlanError::Pending(
                PendingTranscriptBoundary::OodsSampledValues
            ))
        ));
    }

    #[test]
    fn schedule_key_binds_shape_and_segments() {
        let first = test_plan();
        let second = plan_with_interaction_pow_bits(
            ClaimMixShape {
                enable_felts: 2,
                log_size_felts: 2,
                public_data_felts: 3,
            },
            test_pcs(),
            10,
            DynamicTranscriptShape {
                interaction_claim_felts: Some(6),
                oods_sampled_values_felts: Some(7),
            },
            0,
        )
        .unwrap();
        assert_ne!(first.schedule_key(), second.schedule_key());
        assert_eq!(first.schedule_key(), 0x2e69_7865_acf5_0354);
    }

    #[test]
    fn static_claim_encoding_matches_the_reference_mix_order() {
        let claim: CairoClaim = serde_json::from_value(serde_json::json!({
            "public_data": cairo_air::air::PublicData::default(),
            "add_opcode": { "log_size": 4 },
            "memory_id_to_big": { "big_log_sizes": [] }
        }))
        .unwrap();
        let pcs = test_pcs();
        let material = encode_static_transcript_inputs(17, pcs, &claim)
            .unwrap()
            .into_iter()
            .collect::<BTreeMap<_, _>>();

        let mut reference = Blake2sChannel::default();
        reference.mix_felts(&[SecureField::from(17u32)]);
        pcs.mix_into(&mut reference);
        claim.mix_into::<Blake2sMerkleChannel>(&mut reference);

        let mut encoded = Blake2sChannel::default();
        encoded.mix_felts(&felts(&material[&CairoTranscriptInput::ChannelSalt]));
        encoded.mix_felts(&felts(&material[&CairoTranscriptInput::PcsConfig]));
        for semantic in [
            CairoTranscriptInput::ClaimComponentCount,
            CairoTranscriptInput::ClaimEnableBits,
            CairoTranscriptInput::ClaimLogSizes,
            CairoTranscriptInput::ClaimProgramLength,
            CairoTranscriptInput::ClaimPublicData,
        ] {
            encoded.mix_felts(&felts(&material[&semantic]));
        }
        Blake2sMerkleChannel::mix_root(
            &mut encoded,
            root(&material[&CairoTranscriptInput::ClaimOutputRoot]),
        );
        Blake2sMerkleChannel::mix_root(
            &mut encoded,
            root(&material[&CairoTranscriptInput::ClaimProgramRoot]),
        );
        assert_eq!(encoded.digest(), reference.digest());
    }

    #[test]
    fn production_schedule_pins_the_cairo_interaction_pow() {
        assert_eq!(INTERACTION_POW_BITS, 24);
        let plan = plan_with_interaction_pow_bits(
            ClaimMixShape {
                enable_felts: 1,
                log_size_felts: 1,
                public_data_felts: 1,
            },
            test_pcs(),
            10,
            DynamicTranscriptShape {
                interaction_claim_felts: Some(1),
                oods_sampled_values_felts: Some(1),
            },
            INTERACTION_POW_BITS,
        )
        .unwrap();
        assert!(matches!(
            plan.schedule().operations()[11],
            TranscriptOperation::AbsorbPowNonce { pow_bits: 24, .. }
        ));
    }
}
