//! Pinned, non-aliasing ProverInput snapshot used before any fixture write.

use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use bincode::Options;
use sha2::{Digest, Sha256};
use stwo_cairo_adapter::ProverInput;

// 256 MiB keeps the source-only three-copy estimate plus the 128 MiB reserve
// below the exporter's 1 GiB process contract before bincode can allocate.
const MAX_PROVER_INPUT_BYTES: u64 = 256 * 1024 * 1024;
static SNAPSHOT_SERIAL: AtomicU64 = AtomicU64::new(0);

pub fn read_pinned(
    path: &Path,
    expected_sha256: &str,
) -> Result<(ProverInput, String, u64), String> {
    let snapshot = SealedSnapshot::copy(path)?;
    if snapshot.sha256 != expected_sha256 {
        return Err(format!(
            "ProverInput sha256 {} != required {expected_sha256}; refusing to deserialize",
            snapshot.sha256
        ));
    }
    let mut reader =
        BufReader::new(File::open(&snapshot.path).map_err(|error| {
            format!("open sealed snapshot {}: {error}", snapshot.path.display())
        })?);
    let options = bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .allow_trailing_bytes()
        .with_limit(snapshot.bytes);
    let input: ProverInput = options
        .deserialize_from(&mut reader)
        .map_err(|error| format!("deserialize sealed ProverInput: {error}"))?;
    let mut trailing = [0; 1];
    if reader
        .read(&mut trailing)
        .map_err(|error| format!("check sealed ProverInput trailing bytes: {error}"))?
        != 0
    {
        return Err("sealed ProverInput has trailing bytes".into());
    }
    reader
        .seek(SeekFrom::Start(0))
        .map_err(|error| format!("rewind sealed ProverInput: {error}"))?;
    let after = hash_reader(&mut reader)?;
    if after != snapshot.sha256 {
        return Err(format!(
            "sealed ProverInput changed during deserialization: {} -> {after}",
            snapshot.sha256
        ));
    }
    Ok((input, snapshot.sha256.clone(), snapshot.bytes))
}

pub fn reject_aliases(source: &Path, index: &Path, output: &Path) -> Result<(), String> {
    let paths = [
        ("ProverInput", source),
        ("fixture index", index),
        ("oracle output", output),
    ];
    let identities = paths
        .iter()
        .map(|(label, path)| path_identity(path).map(|identity| (*label, identity)))
        .collect::<Result<Vec<_>, _>>()?;
    for left in 0..identities.len() {
        for right in left + 1..identities.len() {
            if identities[left].1.same_file(&identities[right].1) {
                return Err(format!(
                    "{} and {} resolve to the same file",
                    identities[left].0, identities[right].0
                ));
            }
        }
    }
    Ok(())
}

struct SealedSnapshot {
    path: PathBuf,
    sha256: String,
    bytes: u64,
}

impl SealedSnapshot {
    fn copy(source: &Path) -> Result<Self, String> {
        let mut input =
            File::open(source).map_err(|error| format!("open {}: {error}", source.display()))?;
        let metadata = input
            .metadata()
            .map_err(|error| format!("stat {}: {error}", source.display()))?;
        if !metadata.is_file() || metadata.len() == 0 || metadata.len() > MAX_PROVER_INPUT_BYTES {
            return Err(format!(
                "ProverInput {} has invalid byte length {}",
                source.display(),
                metadata.len()
            ));
        }
        let (path, mut output) = create_snapshot_file()?;
        let copied = copy_and_hash(&mut input, &mut output, MAX_PROVER_INPUT_BYTES);
        let (bytes, sha256) = match copied {
            Ok(result) => result,
            Err(error) => {
                drop(output);
                let _ = fs::remove_file(&path);
                return Err(error);
            }
        };
        if let Err(error) = output.sync_all() {
            drop(output);
            let _ = fs::remove_file(&path);
            return Err(format!("sync sealed ProverInput: {error}"));
        }
        if let Err(error) = fs::set_permissions(&path, fs::Permissions::from_mode(0o400)) {
            drop(output);
            let _ = fs::remove_file(&path);
            return Err(format!("seal ProverInput permissions: {error}"));
        }
        Ok(Self {
            path,
            sha256,
            bytes,
        })
    }
}

impl Drop for SealedSnapshot {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn create_snapshot_file() -> Result<(PathBuf, File), String> {
    for _ in 0..32 {
        let serial = SNAPSHOT_SERIAL.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            ".stwo-gpu-lab-prover-input-{}-{serial}",
            std::process::id()
        ));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => return Ok((path, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(format!("create sealed ProverInput snapshot: {error}")),
        }
    }
    Err("cannot allocate unique sealed ProverInput snapshot".into())
}

fn copy_and_hash(
    input: &mut File,
    output: &mut File,
    maximum: u64,
) -> Result<(u64, String), String> {
    let mut hasher = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    let mut bytes = 0u64;
    loop {
        let count = input
            .read(&mut buffer)
            .map_err(|error| format!("read ProverInput: {error}"))?;
        if count == 0 {
            break;
        }
        bytes = bytes
            .checked_add(count as u64)
            .ok_or("ProverInput byte count overflow")?;
        if bytes > maximum {
            return Err(format!("ProverInput exceeds bounded maximum {maximum}"));
        }
        hasher.update(&buffer[..count]);
        output
            .write_all(&buffer[..count])
            .map_err(|error| format!("write sealed ProverInput: {error}"))?;
    }
    Ok((bytes, format!("{:x}", hasher.finalize())))
}

fn hash_reader(reader: &mut impl Read) -> Result<String, String> {
    let mut hasher = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let count = reader
            .read(&mut buffer)
            .map_err(|error| format!("hash snapshot: {error}"))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

struct PathIdentity {
    resolved: PathBuf,
    inode: Option<(u64, u64)>,
}

impl PathIdentity {
    fn same_file(&self, other: &Self) -> bool {
        self.resolved == other.resolved || self.inode.is_some() && self.inode == other.inode
    }
}

fn path_identity(path: &Path) -> Result<PathIdentity, String> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|error| format!("resolve current directory: {error}"))?
            .join(path)
    };
    let resolved = if absolute.exists() {
        fs::canonicalize(&absolute)
            .map_err(|error| format!("resolve {}: {error}", path.display()))?
    } else {
        let parent = absolute
            .parent()
            .ok_or_else(|| format!("{} has no parent", path.display()))?;
        fs::canonicalize(parent)
            .map_err(|error| format!("resolve parent of {}: {error}", path.display()))?
            .join(
                absolute
                    .file_name()
                    .ok_or_else(|| format!("{} has no filename", path.display()))?,
            )
    };
    let inode = fs::metadata(&absolute)
        .ok()
        .map(|metadata| (metadata.dev(), metadata.ino()));
    Ok(PathIdentity { resolved, inode })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_path_and_hardlink_are_rejected() {
        let root = std::env::temp_dir().join(format!("gpu-lab-alias-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let source = root.join("source");
        fs::write(&source, b"source").unwrap();
        let hardlink = root.join("hardlink");
        fs::hard_link(&source, &hardlink).unwrap();
        assert!(reject_aliases(&source, &source, &root.join("out")).is_err());
        assert!(reject_aliases(&source, &root.join("index"), &hardlink).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn source_cap_is_safe_before_deserialization() {
        let estimate =
            crate::producer::estimated_exporter_peak_bytes(MAX_PROVER_INPUT_BYTES, 0, 0).unwrap();
        assert!(estimate <= crate::producer::MAX_EXPORTER_ESTIMATED_PEAK_BYTES);
    }
}
