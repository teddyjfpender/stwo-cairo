//! Bounded, fail-closed output staging for `kernel_emit`.
//!
//! Generated CUDA sources can be multi-megabyte. Keep only compact identities
//! in memory, write each first-seen cache key to a same-filesystem staging
//! directory, and replace the live directory only after the whole run passes.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{self, BufReader, Read};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const COMPARE_BUFFER_BYTES: usize = 64 * 1024;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelEntry {
    pub kind: String,
    pub label: String,
    pub kernel_name: String,
    pub cache_key: u64,
    pub semantic_hash: u64,
    pub file: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SeenKernel {
    kernel_name: String,
    semantic_hash: u64,
    source_len: usize,
    source_digest: u64,
    file: String,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct StagedDiff {
    pub changed: Vec<String>,
    pub unchanged: usize,
    pub stale: Vec<String>,
}

impl StagedDiff {
    pub fn is_clean(&self) -> bool {
        self.changed.is_empty() && self.stale.is_empty()
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PromotionSummary {
    pub changed: usize,
    pub unchanged: usize,
    pub stale: usize,
}

pub struct StagedOutput {
    output_dir: PathBuf,
    staging_dir: PathBuf,
    seen: BTreeMap<u64, SeenKernel>,
    kernels: Vec<KernelEntry>,
    files: BTreeSet<String>,
    staging_owned: bool,
}

impl StagedOutput {
    pub fn new(output_dir: &Path) -> Result<Self, String> {
        let parent = nonempty_parent(output_dir);
        fs::create_dir_all(parent)
            .map_err(|error| format!("create output parent {}: {error}", parent.display()))?;
        let staging_dir = unique_sibling(output_dir, "staging");
        fs::create_dir(&staging_dir).map_err(|error| {
            format!(
                "create staging directory {}: {error}",
                staging_dir.display()
            )
        })?;
        Ok(Self {
            output_dir: output_dir.to_path_buf(),
            staging_dir,
            seen: BTreeMap::new(),
            kernels: Vec::new(),
            files: BTreeSet::new(),
            staging_owned: true,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn stage_kernel(
        &mut self,
        kind: &str,
        label: String,
        kernel_name: String,
        cache_key: u64,
        semantic_hash: u64,
        source: String,
    ) -> Result<bool, String> {
        let source_len = source.len();
        let source_digest = fnv1a64(source.as_bytes());
        if let Some(existing) = self.seen.get(&cache_key) {
            let exact_source = existing.source_len == source_len
                && existing.source_digest == source_digest
                && file_matches_bytes(&self.staging_dir.join(&existing.file), source.as_bytes())
                    .map_err(|error| {
                        format!("compare staged cache key {cache_key:016x}: {error}")
                    })?;
            if existing.kernel_name != kernel_name
                || existing.semantic_hash != semantic_hash
                || !exact_source
            {
                return Err(format!(
                    "AOT cache-key collision {cache_key:016x}: first kernel={} semantic={:016x} \
                     source={:016x}/{}B, later kernel={} semantic={semantic_hash:016x} \
                     source={source_digest:016x}/{source_len}B",
                    existing.kernel_name,
                    existing.semantic_hash,
                    existing.source_digest,
                    existing.source_len,
                    kernel_name,
                ));
            }
            return Ok(false);
        }

        let file = format!("{kind}_{label}_{cache_key:016x}.cu");
        fs::write(self.staging_dir.join(&file), source)
            .map_err(|error| format!("stage {file}: {error}"))?;
        self.seen.insert(
            cache_key,
            SeenKernel {
                kernel_name: kernel_name.clone(),
                semantic_hash,
                source_len,
                source_digest,
                file: file.clone(),
            },
        );
        self.kernels.push(KernelEntry {
            kind: kind.to_owned(),
            label,
            kernel_name,
            cache_key,
            semantic_hash,
            file: file.clone(),
        });
        self.files.insert(file);
        Ok(true)
    }

    pub fn stage_metadata(&mut self, name: &str, content: String) -> Result<(), String> {
        if !self.files.insert(name.to_owned()) {
            return Err(format!("duplicate generated output name {name}"));
        }
        fs::write(self.staging_dir.join(name), content)
            .map_err(|error| format!("stage metadata {name}: {error}"))
    }

    pub fn kernels(&self) -> &[KernelEntry] {
        &self.kernels
    }

    pub fn file_counts(&self) -> (usize, usize) {
        let kernels = self
            .files
            .iter()
            .filter(|file| file.ends_with(".cu"))
            .count();
        (kernels, self.files.len() - kernels)
    }

    pub fn diff(&self) -> Result<StagedDiff, String> {
        let mut changed = Vec::new();
        let mut unchanged = 0usize;
        for name in &self.files {
            let equal = files_equal(&self.staging_dir.join(name), &self.output_dir.join(name))
                .unwrap_or(false);
            if equal {
                unchanged += 1;
            } else {
                changed.push(name.clone());
            }
        }

        let live_entries = collect_relative_entries(&self.output_dir)
            .map_err(|error| format!("inspect {}: {error}", self.output_dir.display()))?;
        let stale = live_entries
            .into_iter()
            .filter(|name| !self.files.contains(name))
            .collect();
        Ok(StagedDiff {
            changed,
            unchanged,
            stale,
        })
    }

    pub fn promote_if_changed(mut self) -> Result<PromotionSummary, String> {
        let diff = self.diff()?;
        let summary = PromotionSummary {
            changed: diff.changed.len(),
            unchanged: diff.unchanged,
            stale: diff.stale.len(),
        };
        if diff.is_clean() {
            return Ok(summary);
        }

        let backup = unique_sibling(&self.output_dir, "backup");
        let had_output = self.output_dir.exists();
        if had_output {
            fs::rename(&self.output_dir, &backup).map_err(|error| {
                format!(
                    "move live output {} to rollback backup {}: {error}",
                    self.output_dir.display(),
                    backup.display()
                )
            })?;
        }
        if let Err(install_error) = fs::rename(&self.staging_dir, &self.output_dir) {
            let restore_error = had_output
                .then(|| fs::rename(&backup, &self.output_dir))
                .and_then(Result::err);
            return Err(match (had_output, restore_error) {
                (true, Some(restore_error)) => format!(
                    "install staged output failed: {install_error}; rollback also failed: \
                     {restore_error}; backup remains at {}",
                    backup.display()
                ),
                (true, None) => format!(
                    "install staged output failed and live output was restored: {install_error}"
                ),
                (false, None) => format!(
                    "install staged output failed; no live output existed or changed: \
                     {install_error}"
                ),
                (false, Some(_)) => unreachable!("no rollback without a live output"),
            });
        }
        self.staging_owned = false;
        if had_output {
            if let Err(error) = fs::remove_dir_all(&backup) {
                eprintln!(
                    "kernel_emit: WARNING installed output but could not remove rollback backup \
                     {}: {error}",
                    backup.display()
                );
            }
        }
        Ok(summary)
    }
}

impl Drop for StagedOutput {
    fn drop(&mut self) {
        if self.staging_owned {
            let _ = fs::remove_dir_all(&self.staging_dir);
        }
    }
}

fn nonempty_parent(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

fn unique_sibling(output_dir: &Path, role: &str) -> PathBuf {
    let name = output_dir
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("generated");
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    nonempty_parent(output_dir).join(format!(
        ".{name}.kernel-emit-{role}-{}-{nonce}",
        std::process::id()
    ))
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

fn file_matches_bytes(path: &Path, expected: &[u8]) -> io::Result<bool> {
    if fs::metadata(path)?.len() != expected.len() as u64 {
        return Ok(false);
    }
    let mut reader = BufReader::new(File::open(path)?);
    let mut offset = 0usize;
    let mut buffer = [0u8; COMPARE_BUFFER_BYTES];
    while offset < expected.len() {
        let read = reader.read(&mut buffer)?;
        if read == 0 || buffer[..read] != expected[offset..offset + read] {
            return Ok(false);
        }
        offset += read;
    }
    Ok(reader.read(&mut buffer)? == 0)
}

fn files_equal(left: &Path, right: &Path) -> io::Result<bool> {
    let left_meta = fs::metadata(left)?;
    let right_meta = fs::metadata(right)?;
    if !left_meta.is_file() || !right_meta.is_file() || left_meta.len() != right_meta.len() {
        return Ok(false);
    }
    let mut left = BufReader::new(File::open(left)?);
    let mut right = BufReader::new(File::open(right)?);
    let mut left_buffer = [0u8; COMPARE_BUFFER_BYTES];
    let mut right_buffer = [0u8; COMPARE_BUFFER_BYTES];
    loop {
        let left_read = left.read(&mut left_buffer)?;
        let right_read = right.read(&mut right_buffer)?;
        if left_read != right_read || left_buffer[..left_read] != right_buffer[..right_read] {
            return Ok(false);
        }
        if left_read == 0 {
            return Ok(true);
        }
    }
}

fn collect_relative_entries(root: &Path) -> io::Result<BTreeSet<String>> {
    let mut entries = BTreeSet::new();
    if !root.exists() {
        return Ok(entries);
    }
    collect_relative_entries_from(root, root, &mut entries)?;
    Ok(entries)
}

fn collect_relative_entries_from(
    root: &Path,
    directory: &Path,
    entries: &mut BTreeSet<String>,
) -> io::Result<()> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        let relative = path.strip_prefix(root).expect("entry descends from root");
        let name = relative.to_string_lossy().replace('\\', "/");
        if path.is_dir() {
            entries.insert(format!("{name}/"));
            collect_relative_entries_from(root, &path, entries)?;
        } else {
            entries.insert(name);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::StagedOutput;

    fn temp_path(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "stwo-kernel-emit-{label}-{}-{nonce}",
            std::process::id()
        ))
    }

    #[test]
    fn exact_duplicate_keeps_the_first_label_deterministically() {
        let root = temp_path("dedup");
        let output = root.join("generated");
        let mut staged = StagedOutput::new(&output).unwrap();
        assert!(staged
            .stage_kernel(
                "constraint",
                "first".to_owned(),
                "kernel".to_owned(),
                7,
                11,
                "source".to_owned(),
            )
            .unwrap());
        assert!(!staged
            .stage_kernel(
                "constraint",
                "later".to_owned(),
                "kernel".to_owned(),
                7,
                11,
                "source".to_owned(),
            )
            .unwrap());
        assert_eq!(staged.kernels().len(), 1);
        assert_eq!(staged.kernels()[0].label, "first");
        assert!(staged.kernels()[0].file.contains("first"));
        drop(staged);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn collision_fails_without_touching_live_output() {
        let root = temp_path("collision");
        let output = root.join("generated");
        fs::create_dir_all(&output).unwrap();
        fs::write(output.join("sentinel"), "live").unwrap();
        let mut staged = StagedOutput::new(&output).unwrap();
        staged
            .stage_kernel(
                "constraint",
                "first".to_owned(),
                "kernel".to_owned(),
                7,
                11,
                "source".to_owned(),
            )
            .unwrap();
        let error = staged
            .stage_kernel(
                "constraint",
                "later".to_owned(),
                "other_kernel".to_owned(),
                7,
                11,
                "different source".to_owned(),
            )
            .unwrap_err();
        assert!(error.contains("cache-key collision"));
        drop(staged);
        assert_eq!(fs::read_to_string(output.join("sentinel")).unwrap(), "live");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn exact_source_check_rejects_staged_bytes_not_admitted_by_digest_metadata() {
        let root = temp_path("exact-source");
        let output = root.join("generated");
        let mut staged = StagedOutput::new(&output).unwrap();
        staged
            .stage_kernel(
                "constraint",
                "first".to_owned(),
                "kernel".to_owned(),
                7,
                11,
                "source".to_owned(),
            )
            .unwrap();
        let staged_file = staged.staging_dir.join(&staged.kernels()[0].file);
        // Same length; `seen` still holds the original source digest. Exact
        // bytes on disk remain a required part of duplicate admission.
        fs::write(staged_file, "sourcf").unwrap();
        let error = staged
            .stage_kernel(
                "constraint",
                "later".to_owned(),
                "kernel".to_owned(),
                7,
                11,
                "source".to_owned(),
            )
            .unwrap_err();
        assert!(error.contains("cache-key collision"));
        drop(staged);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn metadata_name_collision_fails_closed() {
        let root = temp_path("metadata-collision");
        let output = root.join("generated");
        let mut staged = StagedOutput::new(&output).unwrap();
        staged
            .stage_metadata("aot_manifest.json", "first\n".to_owned())
            .unwrap();
        let error = staged
            .stage_metadata("aot_manifest.json", "later\n".to_owned())
            .unwrap_err();
        assert!(error.contains("duplicate generated output name"));
        assert_eq!(
            fs::read_to_string(staged.staging_dir.join("aot_manifest.json")).unwrap(),
            "first\n"
        );
        drop(staged);
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn clean_output_keeps_live_directory_identity_and_removes_staging() {
        use std::os::unix::fs::MetadataExt;

        let root = temp_path("clean");
        let output = root.join("generated");
        fs::create_dir_all(&output).unwrap();
        fs::write(output.join("aot_manifest.json"), "[]\n").unwrap();
        let before = fs::metadata(&output).unwrap();
        let mut staged = StagedOutput::new(&output).unwrap();
        staged
            .stage_metadata("aot_manifest.json", "[]\n".to_owned())
            .unwrap();
        let staging_dir = staged.staging_dir.clone();
        let diff = staged.diff().unwrap();
        assert!(diff.is_clean());
        let summary = staged.promote_if_changed().unwrap();
        let after = fs::metadata(&output).unwrap();
        assert_eq!(summary.changed, 0);
        assert_eq!(summary.unchanged, 1);
        assert_eq!(before.ino(), after.ino());
        assert_eq!(before.modified().unwrap(), after.modified().unwrap());
        assert_eq!(
            fs::read_to_string(output.join("aot_manifest.json")).unwrap(),
            "[]\n"
        );
        assert!(!staging_dir.exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn promotion_replaces_the_complete_directory_and_stale_tree() {
        let root = temp_path("promote");
        let output = root.join("generated");
        fs::create_dir_all(output.join("stale/nested")).unwrap();
        fs::write(output.join("stale/nested/file"), "old").unwrap();
        let mut staged = StagedOutput::new(&output).unwrap();
        staged
            .stage_kernel(
                "witness",
                "kept".to_owned(),
                "kernel".to_owned(),
                9,
                13,
                "new source".to_owned(),
            )
            .unwrap();
        staged
            .stage_metadata("aot_manifest.json", "[]\n".to_owned())
            .unwrap();
        let diff = staged.diff().unwrap();
        assert_eq!(diff.changed.len(), 2);
        assert!(diff.stale.iter().any(|path| path == "stale/"));
        let summary = staged.promote_if_changed().unwrap();
        assert_eq!(summary.changed, 2);
        assert!(!output.join("stale").exists());
        assert!(output.join("aot_manifest.json").is_file());
        assert!(output.join("witness_kept_0000000000000009.cu").is_file());
        let _ = fs::remove_dir_all(root);
    }
}
