use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::process;

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::model::validate_sha256;

pub const M31_ROW_ENCODING: &str = "m31-le-u32-row-major-v1";
const COPY_BUFFER_BYTES: usize = 1024 * 1024;

pub struct ChunkReader {
    path: PathBuf,
    reader: BufReader<File>,
    hasher: Sha256,
    expected_sha256: String,
    remaining: u64,
}

impl ChunkReader {
    pub fn open(
        index_path: &Path,
        relative_path: &str,
        expected_sha256: &str,
        byte_len: u64,
        maximum: u64,
    ) -> Result<Self, String> {
        validate_sha256(expected_sha256, "chunk sha256")?;
        if byte_len > maximum {
            return Err(format!(
                "chunk {relative_path} is {byte_len} bytes; bounded maximum is {maximum}"
            ));
        }
        let path = resolve_content_path(index_path, relative_path, expected_sha256)?;
        let actual_len = fs::metadata(&path)
            .map_err(|error| format!("stat {}: {error}", path.display()))?
            .len();
        if actual_len != byte_len {
            return Err(format!(
                "chunk {} size {actual_len} != declared {byte_len}",
                path.display()
            ));
        }
        let file =
            File::open(&path).map_err(|error| format!("open chunk {}: {error}", path.display()))?;
        Ok(Self {
            path,
            reader: BufReader::new(file),
            hasher: Sha256::new(),
            expected_sha256: expected_sha256.to_owned(),
            remaining: byte_len,
        })
    }

    pub fn read_word(&mut self) -> Result<u32, String> {
        if self.remaining < 4 {
            return Err(format!(
                "chunk {} has no complete word left",
                self.path.display()
            ));
        }
        let mut bytes = [0; 4];
        self.reader
            .read_exact(&mut bytes)
            .map_err(|error| format!("read chunk {}: {error}", self.path.display()))?;
        self.hasher.update(bytes);
        self.remaining -= 4;
        Ok(u32::from_le_bytes(bytes))
    }

    pub fn finish(mut self) -> Result<(), String> {
        if self.remaining != 0 {
            return Err(format!(
                "chunk {} has {} unread bytes",
                self.path.display(),
                self.remaining
            ));
        }
        let mut trailing = [0; 1];
        if self
            .reader
            .read(&mut trailing)
            .map_err(|error| format!("finish chunk {}: {error}", self.path.display()))?
            != 0
        {
            return Err(format!("chunk {} grew while reading", self.path.display()));
        }
        let actual = format!("{:x}", self.hasher.finalize());
        if actual != self.expected_sha256 {
            return Err(format!(
                "chunk {} sha256 {actual} != declared {}",
                self.path.display(),
                self.expected_sha256
            ));
        }
        Ok(())
    }
}

pub struct ChunkWriter {
    root: PathBuf,
    temp: PathBuf,
    writer: BufWriter<File>,
    hasher: Sha256,
    byte_len: u64,
}

pub struct StoredChunk {
    pub relative_path: String,
    pub sha256: String,
    pub byte_len: u64,
}

impl ChunkWriter {
    pub fn create(index_path: &Path, serial: usize) -> Result<Self, String> {
        let artifact_root = index_path.parent().unwrap_or_else(|| Path::new("."));
        ensure_plain_directory(artifact_root, true)?;
        let chunks = artifact_root.join("chunks");
        ensure_plain_directory(&chunks, false)?;
        let root = chunks.join("sha256");
        ensure_plain_directory(&root, false)?;
        let temp = root.join(format!(".tmp-{}-{serial}", process::id()));
        let file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temp)
            .map_err(|error| format!("create chunk temp {}: {error}", temp.display()))?;
        Ok(Self {
            root,
            temp,
            writer: BufWriter::new(file),
            hasher: Sha256::new(),
            byte_len: 0,
        })
    }

    pub fn write_word(&mut self, word: u32) -> Result<(), String> {
        let bytes = word.to_le_bytes();
        self.writer
            .write_all(&bytes)
            .map_err(|error| format!("write chunk {}: {error}", self.temp.display()))?;
        self.hasher.update(bytes);
        self.byte_len = self
            .byte_len
            .checked_add(4)
            .ok_or("output chunk byte count overflow")?;
        Ok(())
    }

    pub fn finish(mut self, index_path: &Path) -> Result<StoredChunk, String> {
        self.writer
            .flush()
            .map_err(|error| format!("flush chunk {}: {error}", self.temp.display()))?;
        self.writer
            .get_ref()
            .sync_all()
            .map_err(|error| format!("sync chunk {}: {error}", self.temp.display()))?;
        let digest = format!("{:x}", self.hasher.clone().finalize());
        let final_path = self.root.join(format!("{digest}.m31le"));
        match fs::hard_link(&self.temp, &final_path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                reject_symlink(&final_path, "existing chunk")?;
                verify_file(&final_path, self.byte_len, &digest)?;
            }
            Err(error) => {
                return Err(format!(
                    "install chunk {} -> {}: {error}",
                    self.temp.display(),
                    final_path.display()
                ));
            }
        }
        fs::remove_file(&self.temp)
            .map_err(|error| format!("remove chunk temp {}: {error}", self.temp.display()))?;
        sync_directory(&self.root)?;
        let parent = index_path.parent().unwrap_or_else(|| Path::new("."));
        let relative = final_path
            .strip_prefix(parent)
            .map_err(|_| "installed chunk escaped artifact directory")?;
        Ok(StoredChunk {
            relative_path: relative.to_string_lossy().into_owned(),
            sha256: digest,
            byte_len: self.byte_len,
        })
    }
}

impl Drop for ChunkWriter {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.temp);
    }
}

pub fn write_json<T: Serialize>(path: Option<&Path>, artifact: &T) -> Result<(), String> {
    let Some(path) = path else {
        let stdout = std::io::stdout();
        let mut output = stdout.lock();
        serde_json::to_writer_pretty(&mut output, artifact)
            .map_err(|error| format!("serialize host oracle: {error}"))?;
        output
            .write_all(b"\n")
            .map_err(|error| format!("write host oracle stdout: {error}"))?;
        return Ok(());
    };
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    ensure_plain_directory(parent, true)?;
    let temp = parent.join(format!(
        ".{}.tmp-{}",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("oracle"),
        process::id()
    ));
    let file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temp)
        .map_err(|error| format!("create {}: {error}", temp.display()))?;
    let mut output = BufWriter::new(file);
    let result = (|| {
        serde_json::to_writer_pretty(&mut output, artifact)
            .map_err(|error| format!("serialize host oracle: {error}"))?;
        output
            .write_all(b"\n")
            .map_err(|error| format!("write {}: {error}", temp.display()))?;
        output
            .flush()
            .map_err(|error| format!("flush {}: {error}", temp.display()))?;
        output
            .get_ref()
            .sync_all()
            .map_err(|error| format!("sync {}: {error}", temp.display()))?;
        fs::rename(&temp, path)
            .map_err(|error| format!("rename {} -> {}: {error}", temp.display(), path.display()))?;
        sync_directory(parent)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

/// Install a small semantic index exactly once. A repeated byte-identical
/// generation is harmless; a different document at the same path is rejected.
pub fn write_immutable_json<T: Serialize>(path: &Path, artifact: &T) -> Result<(), String> {
    let mut bytes = serde_json::to_vec_pretty(artifact)
        .map_err(|error| format!("serialize immutable JSON: {error}"))?;
    bytes.push(b'\n');
    write_immutable_bytes(path, &bytes)
}

pub fn write_immutable_bytes(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if path.exists() {
        return verify_immutable_bytes(path, bytes);
    }
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    ensure_plain_directory(parent, true)?;
    let temp = parent.join(format!(
        ".{}.tmp-{}",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("immutable-json"),
        process::id()
    ));
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temp)
        .map_err(|error| format!("create {}: {error}", temp.display()))?;
    let result = (|| {
        file.write_all(bytes)
            .map_err(|error| format!("write {}: {error}", temp.display()))?;
        file.sync_all()
            .map_err(|error| format!("sync {}: {error}", temp.display()))?;
        match fs::hard_link(&temp, path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                verify_immutable_bytes(path, bytes)
            }
            Err(error) => Err(format!(
                "install immutable JSON {} -> {}: {error}",
                temp.display(),
                path.display()
            )),
        }?;
        sync_directory(parent)
    })();
    let _ = fs::remove_file(&temp);
    result
}

fn verify_immutable_bytes(path: &Path, expected: &[u8]) -> Result<(), String> {
    reject_symlink(path, "immutable artifact")?;
    let actual = fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    if actual != expected {
        return Err(format!(
            "immutable artifact {} already exists with different bytes",
            path.display()
        ));
    }
    Ok(())
}

fn resolve_content_path(
    index_path: &Path,
    relative_path: &str,
    sha256: &str,
) -> Result<PathBuf, String> {
    let relative = Path::new(relative_path);
    if relative.as_os_str().is_empty()
        || relative.is_absolute()
        || relative
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        return Err(format!("unsafe chunk path: {relative_path}"));
    }
    let expected_name = format!("{sha256}.m31le");
    if relative.file_name().and_then(|name| name.to_str()) != Some(&expected_name) {
        return Err(format!(
            "chunk path {relative_path} is not content-addressed by {sha256}"
        ));
    }
    let root = index_path.parent().unwrap_or_else(|| Path::new("."));
    reject_symlink(root, "fixture root")?;
    let mut cursor = root.to_path_buf();
    for component in relative.components() {
        cursor.push(component.as_os_str());
        reject_symlink(&cursor, "chunk path component")?;
    }
    let canonical_root = root
        .canonicalize()
        .map_err(|error| format!("canonicalize {}: {error}", root.display()))?;
    let candidate = root.join(relative);
    let canonical = candidate
        .canonicalize()
        .map_err(|error| format!("canonicalize {}: {error}", candidate.display()))?;
    if !canonical.starts_with(canonical_root) {
        return Err(format!("chunk path escapes fixture root: {relative_path}"));
    }
    Ok(canonical)
}

fn ensure_plain_directory(path: &Path, recursive: bool) -> Result<(), String> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() || !metadata.is_dir() {
                return Err(format!(
                    "directory {} is a symlink or non-directory",
                    path.display()
                ));
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let result = if recursive {
                fs::create_dir_all(path)
            } else {
                fs::create_dir(path)
            };
            result.map_err(|error| format!("create directory {}: {error}", path.display()))?;
            reject_symlink(path, "created directory")?;
        }
        Err(error) => return Err(format!("inspect directory {}: {error}", path.display())),
    }
    Ok(())
}

fn reject_symlink(path: &Path, label: &str) -> Result<(), String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("inspect {label} {}: {error}", path.display()))?;
    if metadata.file_type().is_symlink() {
        return Err(format!("{label} {} is a symlink", path.display()));
    }
    Ok(())
}

fn verify_file(path: &Path, byte_len: u64, sha256: &str) -> Result<(), String> {
    let actual_len = fs::metadata(path)
        .map_err(|error| format!("stat {}: {error}", path.display()))?
        .len();
    if actual_len != byte_len {
        return Err(format!(
            "existing chunk {} size {actual_len} != {byte_len}",
            path.display()
        ));
    }
    let mut input = File::open(path)
        .map_err(|error| format!("open existing chunk {}: {error}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0; COPY_BUFFER_BYTES];
    loop {
        let count = input
            .read(&mut buffer)
            .map_err(|error| format!("hash existing chunk {}: {error}", path.display()))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    let actual = format!("{:x}", hasher.finalize());
    if actual != sha256 {
        return Err(format!(
            "existing chunk {} sha256 {actual} != {sha256}",
            path.display()
        ));
    }
    Ok(())
}

fn sync_directory(path: &Path) -> Result<(), String> {
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| format!("sync directory {}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{write_immutable_json, ChunkWriter};

    #[test]
    fn row_encoding_is_pinned() {
        let words = [1u32, 0x0102_0304];
        let bytes = words
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        assert_eq!(crate::model::sha256_hex(&bytes).len(), 64);
        assert_eq!(&bytes, &[1, 0, 0, 0, 4, 3, 2, 1]);
    }

    #[test]
    fn immutable_json_rejects_semantic_replacement() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "stwo-immutable-json-test-{}-{nonce}",
            std::process::id()
        ));
        let path = root.join("fixture.json");
        write_immutable_json(&path, &serde_json::json!({"value": 1})).unwrap();
        write_immutable_json(&path, &serde_json::json!({"value": 1})).unwrap();
        let error = write_immutable_json(&path, &serde_json::json!({"value": 2}))
            .expect_err("semantic replacement must fail");
        assert!(error.contains("different bytes"));
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn chunk_writer_rejects_symlinked_store() {
        use std::os::unix::fs::symlink;

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "stwo-chunk-symlink-test-{}-{nonce}",
            std::process::id()
        ));
        let outside = root.with_extension("outside");
        fs::create_dir_all(&root).unwrap();
        fs::create_dir_all(&outside).unwrap();
        symlink(&outside, root.join("chunks")).unwrap();
        let error = ChunkWriter::create(&root.join("fixture.json"), 0)
            .err()
            .expect("symlinked chunk store must fail");
        assert!(error.contains("symlink"));
        assert!(!outside.join("sha256").exists());
        let _ = fs::remove_dir_all(root);
        let _ = fs::remove_dir_all(outside);
    }
}
