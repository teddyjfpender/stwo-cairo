use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
use stwo::prover::backend::simd::SimdBackend;
use stwo_cairo_adapter::ProverInput;
use stwo_cairo_dev_utils::utils::get_compiled_cairo_program_path;
use stwo_cairo_prover::prover::{prove_cairo, ProverParameters};
use stwo_cairo_serialize::CairoSerialize;

const REFERENCE_SCHEMA: u32 = 2;
const MAGIC: &[u8; 8] = b"STWOREF2";
const STWO_HEAD_ENV: &str = "STWO_PARITY_REF_STWO_HEAD";
const STWO_WORKTREE_HASH_ENV: &str = "STWO_PARITY_REF_STWO_WORKTREE_HASH";
const STWO_CAIRO_HEAD_ENV: &str = "STWO_PARITY_REF_STWO_CAIRO_HEAD";
const STWO_CAIRO_WORKTREE_HASH_ENV: &str = "STWO_PARITY_REF_STWO_CAIRO_WORKTREE_HASH";
static SOURCE_IDENTITIES: OnceLock<Option<(Vec<u8>, Vec<u8>)>> = OnceLock::new();
static STAGING_ORDINAL: AtomicU64 = AtomicU64::new(0);

pub(crate) fn git_identity(repo: &Path) -> Option<Vec<u8>> {
    let output = |args: &[&str]| -> Option<Vec<u8>> {
        let output = Command::new("git")
            .args(args)
            .current_dir(repo)
            .output()
            .ok()?;
        output.status.success().then_some(output.stdout)
    };
    let append_sized = |identity: &mut Vec<u8>, value: &[u8]| {
        identity.extend_from_slice(&(value.len() as u64).to_le_bytes());
        identity.extend_from_slice(value);
    };
    let mut identity = b"git-source-identity-v2\0".to_vec();
    append_sized(&mut identity, &output(&["rev-parse", "HEAD"])?);
    append_sized(&mut identity, &output(&["diff", "--binary", "HEAD", "--"])?);
    let untracked = output(&["ls-files", "--others", "--exclude-standard", "-z"])?;
    for relative in untracked
        .split(|byte| *byte == 0)
        .filter(|path| !path.is_empty())
    {
        let relative = std::str::from_utf8(relative).ok()?;
        let path = Path::new(relative);
        if !is_source_relevant(path) {
            continue;
        }
        let full_path = repo.join(path);
        let metadata = std::fs::symlink_metadata(&full_path).ok()?;
        if metadata.file_type().is_symlink() {
            identity.extend_from_slice(b"symlink\0");
            append_sized(&mut identity, relative.as_bytes());
            let target = std::fs::read_link(full_path).ok()?;
            append_sized(&mut identity, target.to_str()?.as_bytes());
        } else if metadata.is_file() {
            identity.extend_from_slice(b"file\0");
            append_sized(&mut identity, relative.as_bytes());
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                identity.push(u8::from(metadata.permissions().mode() & 0o111 != 0));
            }
            #[cfg(not(unix))]
            identity.push(0);
            identity.extend_from_slice(blake3::hash(&std::fs::read(full_path).ok()?).as_bytes());
        } else {
            return None;
        }
    }
    Some(identity)
}

pub(crate) fn runner_identity(head: Option<&str>, worktree_hash: Option<&str>) -> Option<Vec<u8>> {
    let head = head?;
    let worktree_hash = worktree_hash?;
    let valid_hex = |value: &str, len: usize| {
        value.len() == len
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    };
    if !valid_hex(head, 40) || !valid_hex(worktree_hash, 64) {
        return None;
    }
    Some(
        [
            b"runner-synced-source-v1\0".as_slice(),
            head.as_bytes(),
            b"\0",
            worktree_hash.as_bytes(),
        ]
        .concat(),
    )
}

pub(crate) fn source_identity(repo: &Path, runner_identity: Option<Vec<u8>>) -> Option<Vec<u8>> {
    if std::fs::symlink_metadata(repo.join(".git")).is_ok() {
        git_identity(repo)
    } else {
        runner_identity
    }
}

fn runner_identity_from_env(head: &str, worktree_hash: &str) -> Option<Vec<u8>> {
    let head = std::env::var(head).ok();
    let worktree_hash = std::env::var(worktree_hash).ok();
    runner_identity(head.as_deref(), worktree_hash.as_deref())
}

fn is_source_relevant(path: &Path) -> bool {
    // Only untracked inputs that can affect compilation/code generation.
    // Benchmark results and other potentially huge artifacts are excluded.
    matches!(
        path.extension().and_then(|extension| extension.to_str()),
        Some("rs" | "cu" | "cuh" | "h" | "c" | "cpp" | "toml" | "lock" | "py" | "sh")
    ) || matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some("build.rs" | "Cargo.toml" | "Cargo.lock")
    )
}

pub(crate) fn key_digest(
    params: &[u8],
    input: &[u8],
    fixture: &[u8],
    stwo_identity: &[u8],
    stwo_cairo_identity: &[u8],
) -> blake3::Hash {
    let mut key = blake3::Hasher::new();
    key.update(b"stwo-cairo-reference-cache\0");
    key.update(&REFERENCE_SCHEMA.to_le_bytes());
    key.update(params);
    key.update(blake3::hash(input).as_bytes());
    key.update(blake3::hash(fixture).as_bytes());
    key.update(blake3::hash(stwo_identity).as_bytes());
    key.update(blake3::hash(stwo_cairo_identity).as_bytes());
    key.finalize()
}

fn cache_path(
    fixture: &str,
    tag: &str,
    input: &ProverInput,
    params: ProverParameters,
) -> Option<PathBuf> {
    let cache_dir = std::env::var_os("STWO_PARITY_REF_CACHE").map(PathBuf::from)?;
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let stwo_cairo = manifest.ancestors().nth(3)?;
    let stwo = stwo_cairo.parent()?.join("stwo");
    let identities = SOURCE_IDENTITIES.get_or_init(|| {
        Some((
            source_identity(
                &stwo,
                runner_identity_from_env(STWO_HEAD_ENV, STWO_WORKTREE_HASH_ENV),
            )?,
            source_identity(
                stwo_cairo,
                runner_identity_from_env(STWO_CAIRO_HEAD_ENV, STWO_CAIRO_WORKTREE_HASH_ENV),
            )?,
        ))
    });
    let identities = identities.as_ref()?;
    let input_bytes = bincode::serialize(input).expect("serialize adapted reference input");
    let fixture_bytes = std::fs::read(get_compiled_cairo_program_path(fixture))
        .expect("read compiled reference fixture");
    let params = format!("{params:?}");
    let key = key_digest(
        params.as_bytes(),
        &input_bytes,
        &fixture_bytes,
        &identities.0,
        &identities.1,
    );
    Some(cache_dir.join(format!(
        "{fixture}-{tag}-v{REFERENCE_SCHEMA}-{}.ref",
        key.to_hex()
    )))
}

pub(crate) fn staging_path(path: &Path) -> PathBuf {
    path.with_extension(format!(
        "ref.tmp.{}.{}",
        std::process::id(),
        STAGING_ORDINAL.fetch_add(1, Ordering::Relaxed)
    ))
}

pub(crate) fn decode(bytes: &[u8]) -> Option<Vec<starknet_ff::FieldElement>> {
    let header = 8 + 8 + 32;
    if bytes.len() < header || &bytes[..8] != MAGIC {
        return None;
    }
    let count = usize::try_from(u64::from_le_bytes(bytes[8..16].try_into().ok()?)).ok()?;
    let payload = &bytes[header..];
    if payload.len() != count.checked_mul(32)? || blake3::hash(payload).as_bytes() != &bytes[16..48]
    {
        return None;
    }
    payload
        .chunks_exact(32)
        .map(|chunk| starknet_ff::FieldElement::from_bytes_be(chunk.try_into().ok()?).ok())
        .collect()
}

pub(crate) fn encode(felts: &[starknet_ff::FieldElement]) -> Vec<u8> {
    let mut payload = Vec::with_capacity(felts.len() * 32);
    for felt in felts {
        payload.extend_from_slice(&felt.to_bytes_be());
    }
    let mut bytes = Vec::with_capacity(48 + payload.len());
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&(felts.len() as u64).to_le_bytes());
    bytes.extend_from_slice(blake3::hash(&payload).as_bytes());
    bytes.extend_from_slice(&payload);
    bytes
}

pub fn cached_reference_felts(
    fixture: &str,
    tag: &str,
    input: ProverInput,
    params: ProverParameters,
) -> Vec<starknet_ff::FieldElement> {
    let path = cache_path(fixture, tag, &input, params);
    if let Some(felts) = path
        .as_ref()
        .and_then(|path| std::fs::read(path).ok())
        .and_then(|bytes| decode(&bytes))
    {
        return felts;
    }
    let felts =
        serialize_felts(&prove_cairo::<SimdBackend, Blake2sMerkleChannel>(input, params).unwrap());
    if let Some(path) = path {
        let _ = std::fs::create_dir_all(path.parent().unwrap());
        let bytes = encode(&felts);
        let staging = staging_path(&path);
        let wrote = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&staging)
            .and_then(|mut file| std::io::Write::write_all(&mut file, &bytes));
        if wrote.is_ok() {
            let _ = std::fs::rename(&staging, path);
        }
    }
    felts
}

pub fn serialize_felts<H>(proof: &cairo_air::CairoProof<H>) -> Vec<starknet_ff::FieldElement>
where
    H: stwo::core::vcs_lifted::merkle_hasher::MerkleHasherLifted,
    H::Hash: CairoSerialize,
{
    let mut felts = Vec::new();
    CairoSerialize::serialize(proof, &mut felts);
    felts
}
