"""Remote stdlib programs for immutable source-generation publication."""

from . import generation_controller_scripts as controller
from . import generation_integrity_scripts as integrity

SUPPORT = r"""
import base64
import ctypes
import errno
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
from pathlib import Path

def git(repo, *args, check=True):
    return subprocess.run(
        ["git", "--no-optional-locks", "-C", str(repo), *args],
        check=check,
        capture_output=True,
        text=True,
        env=git_environment(),
    )

def identity(repo):
    output = subprocess.run(
        [sys.executable, "-c", tree_script, str(repo), excludes],
        check=True,
        capture_output=True,
        text=True,
        env=git_environment(),
    ).stdout
    return json.loads(output)

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def canonical(document):
    return json.dumps(
        document, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode() + b"\n"

def load_overlay(transaction, name, meta, expected):
    if not expected:
        if meta is not None:
            raise RuntimeError(f"unexpected source overlay: {name}")
        return None, {}
    if meta is None:
        raise RuntimeError(f"missing source overlay: {name}")
    archive = Path(meta["path"])
    info = archive.lstat()
    if (
        archive.resolve().parent != transaction / "overlays"
        or archive.name != f"{name}.tar"
        or stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or info.st_size != meta["size"]
        or sha256_file(archive) != meta["sha256"]
    ):
        raise RuntimeError(f"invalid source overlay object: {name}")
    source = tarfile.open(archive, "r:")
    try:
        members = source.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or sorted(names) != sorted(expected):
            raise RuntimeError(f"source overlay member mismatch: {name}")
        for member in members:
            item = expected[member.name]
            if item["kind"] == "file" and member.isfile():
                payload = source.extractfile(member)
                digest, size = hashlib.sha256(), 0
                for block in iter(lambda: payload.read(1024 * 1024), b""):
                    digest.update(block)
                    size += len(block)
                content_sha = digest.hexdigest()
            elif item["kind"] == "symlink" and member.issym():
                payload = member.linkname.encode()
                size, content_sha = len(payload), hashlib.sha256(payload).hexdigest()
            else:
                raise RuntimeError(f"source overlay kind mismatch: {name}/{member.name}")
            if (
                member.mode != item["mode"]
                or size != item["size"]
                or content_sha != item["sha256"]
            ):
                raise RuntimeError(f"source overlay identity mismatch: {name}/{member.name}")
        if sha256_file(archive) != meta["sha256"]:
            raise RuntimeError(f"source overlay raced while reading: {name}")
        return source, {member.name: member for member in members}
    except BaseException:
        source.close()
        raise

def safe_parent(root, rel):
    cursor = root
    for part in Path(rel).parts[:-1]:
        cursor = cursor / part
        if os.path.lexists(cursor):
            info = cursor.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise RuntimeError(f"unsafe destination parent: {cursor}")
        else:
            cursor.mkdir()

def preserve(path, quarantine, rel):
    if not os.path.lexists(path):
        return
    quarantine.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256(rel.encode()).hexdigest() + "-" + os.urandom(16).hex()
    if not rename_noreplace(path, quarantine / name):
        raise RuntimeError("quarantine collision while preserving staging bytes")

def write_once(path, payload, publisher_uid, publisher_gid, mode=0o400):
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    except FileExistsError:
        info = path.lstat()
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != publisher_uid
            or info.st_gid != publisher_gid
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != mode
            or path.read_bytes() != payload
        ):
            raise RuntimeError(f"immutable controller object mismatch: {path}")
        return
    with os.fdopen(fd, "wb") as output:
        output.write(payload)
        os.fchmod(output.fileno(), mode)
        output.flush()
        os.fsync(output.fileno())
        info = os.fstat(output.fileno())
        if info.st_uid != publisher_uid or info.st_gid != publisher_gid:
            raise RuntimeError(f"controller object has the wrong publisher: {path}")
""" + controller.SUPPORT + integrity.SUPPORT

PREPARE = SUPPORT + r"""
publisher_uid, publisher_gid = require_publisher(sys.argv[1], sys.argv[2])
requested_root = Path(sys.argv[3])
token = sys.argv[4]
estimated_bytes = int(sys.argv[5])
estimated_entries = int(sys.argv[6])
if (
    requested_root.is_symlink()
    or not re.fullmatch(r"[0-9a-f]{64}", token)
    or not 0 <= estimated_bytes <= 40 * 1024**3
    or not 0 <= estimated_entries <= 1_000_000
):
    raise RuntimeError("unsafe source-generation root or transaction token")
requested_root.mkdir(mode=0o755, parents=True, exist_ok=True)
root = requested_root.resolve()
if str(root) != str(requested_root):
    raise RuntimeError("source-generation root must already be canonical")
validate_controller_root(root, publisher_uid, publisher_gid)
transactions = root / "transactions"
owned_directory(transactions, publisher_uid, publisher_gid)
existing_transactions = list(transactions.iterdir())
if existing_transactions:
    raise RuntimeError(
        "an incomplete source transaction requires bounded failure reconciliation"
    )
free_bytes = os.statvfs(root).f_bavail * os.statvfs(root).f_frsize
if free_bytes < 20 * 1024**3 + estimated_bytes:
    raise RuntimeError(
        "source transaction estimate must leave 20 GiB free on the 100 GiB volume"
    )
failures = root / "failures"
owned_directory(failures, publisher_uid, publisher_gid)
failure_manifests = list(failures.iterdir())
if len(failure_manifests) >= 256:
    raise RuntimeError("failed-transaction manifest bound reached (256)")
for manifest in failure_manifests:
    info = manifest.lstat()
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or info.st_uid != publisher_uid
        or info.st_gid != publisher_gid
        or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) != 0o444
        or info.st_size > 16 * 1024
    ):
        raise RuntimeError("failed-transaction evidence store is invalid or unbounded")
transaction = transactions / token
transaction.mkdir(mode=0o700)
protected_directory(transaction, publisher_uid, publisher_gid, 0o700)
(transaction / "bundles").mkdir()
(transaction / "overlays").mkdir()
(transaction / "quarantine").mkdir()
print(f"LABCTL_SOURCE_TRANSACTION={transaction}")
"""

STAGE = SUPPORT + r"""
publisher_uid, publisher_gid = require_publisher(sys.argv[1], sys.argv[2])
requested_transaction = Path(sys.argv[3])
transaction = requested_transaction.resolve(strict=True)
if requested_transaction.is_symlink() or str(transaction) != str(requested_transaction):
    raise RuntimeError("source transaction must already be canonical")
if transaction.parent.name != "transactions" or not re.fullmatch(r"[0-9a-f]{64}", transaction.name):
    raise RuntimeError("invalid source transaction path")
validate_controller_root(transaction.parent.parent, publisher_uid, publisher_gid)
protected_directory(transaction, publisher_uid, publisher_gid, 0o700)
seeds = json.loads(base64.b64decode(sys.argv[4], validate=True))
seed_identities = json.loads(base64.b64decode(sys.argv[5], validate=True))
bundles = json.loads(base64.b64decode(sys.argv[6], validate=True))
targets = json.loads(base64.b64decode(sys.argv[7], validate=True))
tree_script = base64.b64decode(sys.argv[8], validate=True).decode()
excludes = sys.argv[9]
staging = transaction / "staging"
staging.mkdir(mode=0o700)
(staging / "repos").mkdir()
for name in sorted(seeds):
    seed = Path(seeds[name]).resolve(strict=True)
    if identity(seed) != seed_identities[name] or not index_clean(seed):
        raise RuntimeError(f"seed changed before staging: {name}")
    target = targets[name]
    bundle_meta = bundles.get(name)
    bundle = None
    if bundle_meta:
        bundle = Path(bundle_meta["path"])
        info = bundle.lstat()
        if (
            bundle.resolve().parent != transaction / "bundles"
            or bundle.name != f"{name}.bundle"
            or stat.S_ISLNK(info.st_mode)
            or not stat.S_ISREG(info.st_mode)
            or info.st_size != bundle_meta["size"]
            or sha256_file(bundle) != bundle_meta["sha256"]
        ):
            raise RuntimeError(f"invalid source bundle: {name}")
    repo = staging / "repos" / name
    subprocess.run(
        [
            "git", "--no-optional-locks", "clone", "--local", "--no-hardlinks", "--no-checkout",
            str(seed), str(repo),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=git_environment(),
    )
    if bundle:
        git(repo, "bundle", "verify", str(bundle))
        heads = git(repo, "bundle", "list-heads", str(bundle)).stdout.splitlines()
        if heads != [f"{target} HEAD"]:
            raise RuntimeError(f"bundle target mismatch: {name}")
        git(repo, "fetch", "--no-tags", str(bundle), "HEAD")
        if git(repo, "rev-parse", "FETCH_HEAD^{commit}").stdout.strip() != target:
            raise RuntimeError(f"fetched target mismatch: {name}")
    git(repo, "cat-file", "-e", f"{target}^{{commit}}")
    git(repo, "-c", "advice.detachedHead=false", "checkout", "--detach", target)
    staged_identity = identity(repo)
    if (
        staged_identity["head"] != target
        or staged_identity["entries"]
        or not index_clean(repo)
    ):
        raise RuntimeError(f"staged clone is not exact and clean: {name}")
    (transaction / "quarantine" / name).mkdir()
    if identity(seed) != seed_identities[name] or not index_clean(seed):
        raise RuntimeError(f"seed changed during staging: {name}")
print(f"LABCTL_SOURCE_STAGING={staging}")
"""

FINALIZE = SUPPORT + r"""
publisher_uid, publisher_gid = require_publisher(sys.argv[1], sys.argv[2])
requested_root = Path(sys.argv[3])
root = requested_root.resolve()
requested_transaction = Path(sys.argv[4])
transaction = requested_transaction.resolve(strict=True)
seeds = json.loads(base64.b64decode(sys.argv[5], validate=True))
seed_identities = json.loads(base64.b64decode(sys.argv[6], validate=True))
local_identities = json.loads(base64.b64decode(sys.argv[7], validate=True))
overlays = json.loads(base64.b64decode(sys.argv[8], validate=True))
generation_payload = base64.b64decode(sys.argv[9], validate=True)
generation_sha = sys.argv[10]
expected_current = sys.argv[11] or None
tree_script = base64.b64decode(sys.argv[12], validate=True).decode()
excludes = sys.argv[13]
if requested_root.is_symlink() or str(root) != str(requested_root):
    raise RuntimeError("source-generation root must already be canonical")
if requested_transaction.is_symlink() or str(transaction) != str(requested_transaction):
    raise RuntimeError("source transaction must already be canonical")
validate_controller_root(root, publisher_uid, publisher_gid)
protected_directory(transaction, publisher_uid, publisher_gid, 0o700)
if transaction.parent != root / "transactions" or not re.fullmatch(r"[0-9a-f]{64}", transaction.name):
    raise RuntimeError("source transaction is outside the controller root")
if not re.fullmatch(r"[0-9a-f]{64}", generation_sha):
    raise RuntimeError("invalid source-generation digest")
if hashlib.sha256(generation_payload).hexdigest() != generation_sha:
    raise RuntimeError("generation document digest mismatch")
staging = transaction / "staging"
for name in sorted(seeds):
    seed = Path(seeds[name]).resolve(strict=True)
    repo = staging / "repos" / name
    expected = local_identities[name]
    if identity(seed) != seed_identities[name] or not index_clean(seed):
        raise RuntimeError(f"seed changed before finalize: {name}")
    clean = identity(repo)
    if clean["head"] != expected["head"] or clean["entries"] or not index_clean(repo):
        raise RuntimeError(f"staging changed after clean precheck: {name}")
    present = {
        item["path"]: item for item in expected["entries"]
        if item["kind"] != "deleted"
    }
    overlay, overlay_members = load_overlay(
        transaction, name, overlays.get(name), present
    )
    quarantine = transaction / "quarantine" / name
    for item in expected["entries"]:
        rel = item["path"]
        destination = repo / rel
        safe_parent(repo, rel)
        preserve(destination, quarantine, rel)
        if item["kind"] == "file":
            descriptor = os.open(
                destination,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                item["mode"],
            )
            with os.fdopen(descriptor, "wb") as output:
                payload = overlay.extractfile(overlay_members[rel])
                digest, size = hashlib.sha256(), 0
                for block in iter(lambda: payload.read(1024 * 1024), b""):
                    output.write(block)
                    digest.update(block)
                    size += len(block)
                output.flush()
                os.fsync(output.fileno())
            if size != item["size"] or digest.hexdigest() != item["sha256"]:
                raise RuntimeError(f"source overlay raced during install: {name}/{rel}")
        elif item["kind"] == "symlink":
            os.symlink(overlay_members[rel].linkname, destination)
        elif item["kind"] != "deleted":
            raise RuntimeError(f"unsupported generation entry: {item}")
    if overlay:
        overlay.close()
    if identity(repo) != expected or not index_clean(repo):
        raise RuntimeError(f"final staging identity mismatch: {name}")
    if identity(seed) != seed_identities[name] or not index_clean(seed):
        raise RuntimeError(f"seed changed during finalize: {name}")
write_once(
    staging / "GENERATION.json", generation_payload, publisher_uid, publisher_gid
)
for name, expected in local_identities.items():
    if identity(staging / "repos" / name) != expected or not index_clean(staging / "repos" / name):
        raise RuntimeError(f"staging raced before publication: {name}")
freeze_generation(
    staging, local_identities, generation_payload, publisher_uid, publisher_gid
)
final_root = root / "sha256"
owned_directory(final_root, publisher_uid, publisher_gid)
final = final_root / generation_sha
if final.exists():
    if final.is_symlink():
        raise RuntimeError("existing generation path is not exact")
    validate_generation(
        final, local_identities, generation_payload, publisher_uid, publisher_gid
    )
else:
    if not rename_noreplace(staging, final):
        validate_generation(
            final, local_identities, generation_payload, publisher_uid, publisher_gid
        )
validate_generation(
    final, local_identities, generation_payload, publisher_uid, publisher_gid
)
repositories = []
for name in sorted(local_identities):
    item = local_identities[name]
    repositories.append({
        "head": item["head"],
        "name": name,
        "path": str(final / "repos" / name),
        "tree_content_sha256": item["tree_content_sha256"],
        "worktree_identity_sha256": item["identity_sha256"],
    })
pointer_document = {
    "generation_path": str(final),
    "generation_sha256": generation_sha,
    "repositories": repositories,
    "schema_version": "stwo.gpu-lab.source-pointer.v1",
}
pointer_payload = canonical(pointer_document)
pointer_sha = hashlib.sha256(pointer_payload).hexdigest()
manifests = root / "manifests"
owned_directory(manifests, publisher_uid, publisher_gid)
pointer_manifest = manifests / f"{pointer_sha}.json"
write_once(pointer_manifest, pointer_payload, publisher_uid, publisher_gid)
current = root / "CURRENT"
actual_current = current_pointer_target(
    root, publisher_uid, publisher_gid, validate=False
)
if actual_current != expected_current:
    raise RuntimeError("source pointer changed after precheck")
if actual_current is not None:
    validate_pointer_object(root, actual_current, publisher_uid, publisher_gid)
target = f"manifests/{pointer_sha}.json"
temporary = root / f".CURRENT-{transaction.name}"
os.symlink(target, temporary)
os.replace(temporary, current)
if not current.is_symlink() or os.readlink(current) != target:
    raise RuntimeError("atomic source-pointer swap was not retained")
if current.lstat().st_uid != publisher_uid or current.lstat().st_gid != publisher_gid:
    raise RuntimeError("atomic source pointer has the wrong publisher owner")
validate_pointer_object(root, target, publisher_uid, publisher_gid)
validate_generation(
    final, local_identities, generation_payload, publisher_uid, publisher_gid
)
print(f"LABCTL_SOURCE_GENERATION={final}")
print(f"LABCTL_SOURCE_GENERATION_SHA256={generation_sha}")
print(f"LABCTL_SOURCE_POINTER={current}")
print(f"LABCTL_SOURCE_POINTER_TARGET={target}")
print(f"LABCTL_SOURCE_POINTER_SHA256={pointer_sha}")
"""

ATTEST = SUPPORT + r"""
publisher_uid, publisher_gid = require_publisher(sys.argv[1], sys.argv[2])
requested_root = Path(sys.argv[3])
root = requested_root.resolve(strict=True)
expected = json.loads(base64.b64decode(sys.argv[4], validate=True))
local_identities = json.loads(base64.b64decode(sys.argv[5], validate=True))
tree_script = base64.b64decode(sys.argv[6], validate=True).decode()
excludes = sys.argv[7]
if requested_root.is_symlink() or str(root) != str(requested_root):
    raise RuntimeError("source-generation root must already be canonical")
validate_controller_root(root, publisher_uid, publisher_gid)
target = current_pointer_target(root, publisher_uid, publisher_gid)
if target != expected["pointer_target"]:
    raise RuntimeError("active source pointer changed")
manifest = root / expected["pointer_target"]
payload = manifest.read_bytes()
if hashlib.sha256(payload).hexdigest() != expected["pointer_sha256"]:
    raise RuntimeError("active source-pointer manifest digest mismatch")
document = json.loads(payload)
generation = Path(expected["generation_path"])
repositories = []
for name in sorted(local_identities):
    item = local_identities[name]
    repositories.append({
        "head": item["head"],
        "name": name,
        "path": str(generation / "repos" / name),
        "tree_content_sha256": item["tree_content_sha256"],
        "worktree_identity_sha256": item["identity_sha256"],
    })
expected_document = {
    "generation_path": str(generation),
    "generation_sha256": expected["generation_sha256"],
    "repositories": repositories,
    "schema_version": "stwo.gpu-lab.source-pointer.v1",
}
if document != expected_document:
    raise RuntimeError("active source-pointer manifest target mismatch")
generation_payload = (generation / "GENERATION.json").read_bytes()
if hashlib.sha256(generation_payload).hexdigest() != expected["generation_sha256"]:
    raise RuntimeError("active generation document digest mismatch")
validate_generation(
    generation, local_identities, generation_payload, publisher_uid, publisher_gid
)
print(f"LABCTL_SOURCE_ATTESTED={expected['generation_sha256']}")
"""

CURRENT = SUPPORT + r"""
publisher_uid, publisher_gid = require_publisher(sys.argv[1], sys.argv[2])
requested_root = Path(sys.argv[3])
if not os.path.lexists(requested_root):
    print("LABCTL_SOURCE_CURRENT=NONE")
    sys.exit(0)
root = requested_root.resolve(strict=True)
if requested_root.is_symlink() or str(root) != str(requested_root):
    raise RuntimeError("source-generation root must already be canonical")
validate_controller_root(root, publisher_uid, publisher_gid)
target = current_pointer_target(root, publisher_uid, publisher_gid)
print(f"LABCTL_SOURCE_CURRENT={target or 'NONE'}")
"""
