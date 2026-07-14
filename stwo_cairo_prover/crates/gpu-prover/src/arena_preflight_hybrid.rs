//! Exact host-side accounting and topology-fixture export for the hybrid numerator schedule.

use std::mem::size_of;
use std::path::Path;

use blake3::Hasher;
use stwo::core::circle::CirclePoint;
use stwo::core::fields::qm31::SecureField;
use stwo_backend_cuda::{
    quotient_numerator_hybrid_plan, QuotientNumeratorColumnTopology,
    QuotientNumeratorSourceKind, QuotientNumeratorWorkspaceConfig,
};
use stwo_cairo_gpu_prover::arena_plan::PlannedQuotientNumeratorWorkspace;

const FIXTURE_SCHEMA: &str = "stwo.quotient_numerator.topology_fixture";
const FIXTURE_VERSION: u32 = 1;
const DIGEST_ALGORITHM: &str = "blake3";
const DIGEST_ENCODING: &str = "tag-u32le-payload-u64le-scalars-le.v1";
const DIGEST_DOMAIN: &[u8] = b"stwo.quotient-numerator.topology-fixture.v1\0";

/// Writes only the address-free config and exact canonical topology. Prepared
/// requirements and schedules are deliberately absent so every consumer must
/// rebuild them with the backend under test.
pub fn export_fixture(
    workspace: &PlannedQuotientNumeratorWorkspace,
    path: impl AsRef<Path>,
) -> Result<(), String> {
    let topologies = topologies(workspace);
    let fixture = fixture_json(
        workspace.config,
        &topologies,
        workspace.requirements.input_sample_count,
    );
    let mut encoded = serde_json::to_vec_pretty(&fixture)
        .map_err(|error| format!("serialize quotient topology fixture: {error}"))?;
    encoded.push(b'\n');
    std::fs::write(path.as_ref(), encoded).map_err(|error| {
        format!(
            "write quotient topology fixture {}: {error}",
            path.as_ref().display()
        )
    })
}

pub fn export_requested(workspace: &PlannedQuotientNumeratorWorkspace) -> Result<(), String> {
    let mut args = std::env::args();
    while let Some(argument) = args.next() {
        if argument == "--quotient-topology-fixture-output" {
            let path = args.next().ok_or_else(|| {
                "--quotient-topology-fixture-output requires a path".to_owned()
            })?;
            return export_fixture(workspace, path);
        }
    }
    Ok(())
}

pub fn json(workspace: &PlannedQuotientNumeratorWorkspace) -> serde_json::Value {
    let topologies = topologies(workspace);
    let plan = match quotient_numerator_hybrid_plan(workspace.config, &topologies) {
        Ok(plan) => plan,
        Err(error) => {
            return serde_json::json!({
                "eligible": false,
                "error": error.to_string(),
                "scope": "modeled logical output traffic; not HBM or runtime",
            });
        }
    };
    if plan.requirements() != &workspace.requirements {
        return serde_json::json!({
            "eligible": false,
            "error": "hybrid and resident workspace requirements differ",
            "scope": "modeled logical output traffic; not HBM or runtime",
        });
    }

    let report = plan.report();
    let Some(saved) = report
        .legacy_logical_output_bytes
        .checked_sub(report.hybrid_logical_output_bytes)
    else {
        return serde_json::json!({
            "eligible": false,
            "error": "hybrid traffic exceeds its legacy comparator",
            "scope": "modeled logical output traffic; not HBM or runtime",
        });
    };
    serde_json::json!({
        "eligible": true,
        "eligible_groups": report.eligible_group_count,
        "legacy_groups": report.legacy_group_count,
        "eligible_output_rows": report.eligible_output_rows,
        "legacy_output_rows": report.legacy_output_rows,
        "legacy_batches": report.legacy_batch_count,
        "legacy_logical_output_bytes": report.legacy_logical_output_bytes,
        "hybrid_logical_output_bytes": report.hybrid_logical_output_bytes,
        "saved_logical_output_bytes": saved,
        "reduction_fraction": saved as f64 / report.legacy_logical_output_bytes as f64,
        "traffic_reduction_ratio": report.legacy_logical_output_bytes as f64
            / report.hybrid_logical_output_bytes as f64,
        "packed_term_words": plan.packed_terms().len(),
        "packed_group_offset_words": plan.packed_group_offsets().len(),
        "scope": "modeled logical output traffic; not HBM or runtime",
    })
}

fn topologies(
    workspace: &PlannedQuotientNumeratorWorkspace,
) -> Vec<QuotientNumeratorColumnTopology> {
    workspace
        .columns
        .iter()
        .map(|column| column.topology.clone())
        .collect()
}

fn fixture_json(
    config: QuotientNumeratorWorkspaceConfig,
    columns: &[QuotientNumeratorColumnTopology],
    input_sample_count: usize,
) -> serde_json::Value {
    let sample_count = columns
        .iter()
        .map(|column| column.samples.len())
        .sum::<usize>();
    let digest = fixture_digest(config, columns, input_sample_count);
    let columns = columns
        .iter()
        .enumerate()
        .map(|(column_index, column)| {
            let samples = column
                .samples
                .iter()
                .enumerate()
                .map(|(sample_index, sample)| {
                    serde_json::json!({
                        "sample_index": sample_index,
                        "input_index": sample.input_index,
                        "shape_point_m31": point_limbs(sample.shape_point),
                    })
                })
                .collect::<Vec<_>>();
            serde_json::json!({
                "column_index": column_index,
                "coefficient_log_size": column.coefficient_log_size,
                "source_kind": source_kind(column.source_kind),
                "sample_count": samples.len(),
                "samples": samples,
            })
        })
        .collect::<Vec<_>>();
    serde_json::json!({
        "schema": FIXTURE_SCHEMA,
        "version": FIXTURE_VERSION,
        "digest": {
            "algorithm": DIGEST_ALGORITHM,
            "encoding": DIGEST_ENCODING,
            "hex": digest.to_hex().to_string(),
        },
        "config": {
            "lifting_log_size": config.lifting_log_size,
            "log_blowup_factor": config.log_blowup_factor,
            "max_lde_tile_words": config.max_lde_tile_words,
        },
        "column_count": columns.len(),
        "sample_count": sample_count,
        "input_sample_count": input_sample_count,
        "columns": columns,
    })
}

fn fixture_digest(
    config: QuotientNumeratorWorkspaceConfig,
    columns: &[QuotientNumeratorColumnTopology],
    input_sample_count: usize,
) -> blake3::Hash {
    let sample_count = columns
        .iter()
        .map(|column| column.samples.len())
        .sum::<usize>();
    let mut digest = FramedDigest::new();
    digest.bytes("schema", FIXTURE_SCHEMA.as_bytes());
    digest.u32("version", FIXTURE_VERSION);
    digest.bytes("digest.algorithm", DIGEST_ALGORITHM.as_bytes());
    digest.bytes("digest.encoding", DIGEST_ENCODING.as_bytes());
    digest.u32("config.lifting_log_size", config.lifting_log_size);
    digest.u32("config.log_blowup_factor", config.log_blowup_factor);
    digest.u64("config.max_lde_tile_words", as_u64(config.max_lde_tile_words));
    digest.u64("column_count", as_u64(columns.len()));
    digest.u64("sample_count", as_u64(sample_count));
    digest.u64("input_sample_count", as_u64(input_sample_count));
    for (column_index, column) in columns.iter().enumerate() {
        digest.u64("column.index", as_u64(column_index));
        digest.u32("column.coefficient_log_size", column.coefficient_log_size);
        digest.bytes("column.source_kind", source_kind(column.source_kind).as_bytes());
        digest.u64("column.sample_count", as_u64(column.samples.len()));
        for (sample_index, sample) in column.samples.iter().enumerate() {
            digest.u64("sample.index", as_u64(sample_index));
            digest.u32("sample.input_index", sample.input_index);
            let limbs = point_limbs(sample.shape_point);
            let mut encoded = [0u8; 8 * size_of::<u32>()];
            for (destination, limb) in encoded.chunks_exact_mut(4).zip(limbs) {
                destination.copy_from_slice(&limb.to_le_bytes());
            }
            digest.bytes("sample.shape_point_m31", &encoded);
        }
    }
    digest.finish()
}

fn point_limbs(point: CirclePoint<SecureField>) -> [u32; 8] {
    let mut limbs = [0; 8];
    for (destination, coordinate) in limbs[..4].iter_mut().zip(point.x.to_m31_array()) {
        *destination = coordinate.0;
    }
    for (destination, coordinate) in limbs[4..].iter_mut().zip(point.y.to_m31_array()) {
        *destination = coordinate.0;
    }
    limbs
}

fn source_kind(kind: QuotientNumeratorSourceKind) -> &'static str {
    match kind {
        QuotientNumeratorSourceKind::Evaluation => "evaluation",
        QuotientNumeratorSourceKind::Coefficients => "coefficients",
    }
}

fn as_u64(value: usize) -> u64 {
    u64::try_from(value).expect("topology fixture count exceeds u64")
}

struct FramedDigest(Hasher);

impl FramedDigest {
    fn new() -> Self {
        let mut hasher = Hasher::new();
        hasher.update(DIGEST_DOMAIN);
        Self(hasher)
    }

    fn bytes(&mut self, tag: &str, payload: &[u8]) {
        self.0.update(
            &u32::try_from(tag.len())
                .expect("fixture digest tag exceeds u32")
                .to_le_bytes(),
        );
        self.0.update(tag.as_bytes());
        self.0.update(
            &u64::try_from(payload.len())
                .expect("fixture digest payload exceeds u64")
                .to_le_bytes(),
        );
        self.0.update(payload);
    }

    fn u32(&mut self, tag: &str, value: u32) {
        self.bytes(tag, &value.to_le_bytes());
    }

    fn u64(&mut self, tag: &str, value: u64) {
        self.bytes(tag, &value.to_le_bytes());
    }

    fn finish(self) -> blake3::Hash {
        self.0.finalize()
    }
}
