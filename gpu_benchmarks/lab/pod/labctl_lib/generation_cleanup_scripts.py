"""Bounded cleanup programs for completed and failed source transactions."""

from .generation_scripts import SUPPORT


MAX_TRANSACTION_BYTES = 40 * 1024**3
MAX_TRANSACTION_ENTRIES = 1_000_000
MIN_FREE_BYTES = 20 * 1024**3
MAX_FAILURE_MANIFESTS = 256

_CLEANUP = r"""
import shutil

MAX_BYTES = 40 * 1024**3
MAX_ENTRIES = 1_000_000
FAILURE_POLICY = {
    "max_transaction_bytes": MAX_BYTES,
    "max_transaction_entries": MAX_ENTRIES,
    "min_free_bytes_before_next_transaction": 20 * 1024**3,
}
FAILURE_SCHEMA = "stwo.gpu-lab.source-failure.v1"

def checked_transaction(root, raw, publisher_uid, publisher_gid):
    requested = Path(raw)
    transaction = requested.resolve(strict=True)
    if (
        requested.is_symlink()
        or str(requested) != str(transaction)
        or transaction.parent != root / "transactions"
        or not re.fullmatch(r"[0-9a-f]{64}", transaction.name)
    ):
        raise RuntimeError("unsafe source transaction cleanup path")
    protected_directory(transaction, publisher_uid, publisher_gid, 0o700)
    directories, regular, links, _ = nodes(transaction)
    for path in directories + regular + links:
        require_publisher_owner(path, path.lstat(), publisher_uid, publisher_gid)
    summary = inventory(transaction)
    if (
        summary["bytes"] > MAX_BYTES
        or summary["entries"] > MAX_ENTRIES
        or summary["hardlinks"]
        or summary["special"]
    ):
        raise RuntimeError(f"transaction exceeds safe cleanup bounds: {summary}")
    return transaction, summary

def remove_transaction(transaction):
    directories, regular, _, _ = nodes(transaction)
    for path in regular:
        os.chmod(path, 0o600)
    for directory in directories:
        os.chmod(directory, 0o700)
    shutil.rmtree(transaction)
    if transaction.exists() or transaction.is_symlink():
        raise RuntimeError("source transaction cleanup was incomplete")

def validate_failure_manifest(
    manifest, transaction, summary, publisher_uid, publisher_gid,
    allow_partial=False,
):
    info = manifest.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != publisher_uid
        or info.st_gid != publisher_gid
        or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) != 0o444
        or info.st_size > 16 * 1024
    ):
        raise RuntimeError("existing failure manifest is mutable or invalid")
    payload = manifest.read_bytes()
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("existing failure manifest is not canonical JSON") from error
    if canonical(document) != payload or not isinstance(document, dict):
        raise RuntimeError("existing failure manifest is not canonical JSON")
    base_keys = {
        "cleanup_policy", "error", "schema_version", "transaction",
        "transaction_bytes", "transaction_entries",
    }
    keys = set(document)
    if keys not in (base_keys, base_keys | {"error_sha256"}):
        raise RuntimeError("existing failure manifest has an invalid schema")
    error = document.get("error")
    if not isinstance(error, str) or len(error) > 2048:
        raise RuntimeError("existing failure manifest has an invalid error")
    if "error_sha256" in document:
        if document["error_sha256"] != hashlib.sha256(error.encode()).hexdigest():
            raise RuntimeError("existing failure manifest error digest mismatch")
    elif error != "reconciled incomplete transaction after controller interruption":
        raise RuntimeError("existing reconciliation manifest has an invalid error")
    original_bytes = document.get("transaction_bytes")
    original_entries = document.get("transaction_entries")
    if (
        document.get("cleanup_policy") != FAILURE_POLICY
        or document.get("schema_version") != FAILURE_SCHEMA
        or document.get("transaction") != str(transaction)
        or type(original_bytes) is not int
        or not 0 <= original_bytes <= MAX_BYTES
        or type(original_entries) is not int
        or not 1 <= original_entries <= MAX_ENTRIES
        or summary["bytes"] > original_bytes
        or summary["entries"] > original_entries
        or (
            not allow_partial
            and (
                summary["bytes"] != original_bytes
                or summary["entries"] != original_entries
            )
        )
    ):
        raise RuntimeError("existing failure manifest does not bind this transaction")
    return payload
"""


SUCCESS = SUPPORT + _CLEANUP + r"""
publisher_uid, publisher_gid = require_publisher(sys.argv[1], sys.argv[2])
requested_root = Path(sys.argv[3])
root = requested_root.resolve(strict=True)
if requested_root.is_symlink() or str(root) != str(requested_root):
    raise RuntimeError("source-generation root must already be canonical")
validate_controller_root(root, publisher_uid, publisher_gid)
transaction, summary = checked_transaction(
    root, sys.argv[4], publisher_uid, publisher_gid
)
publication = json.loads(base64.b64decode(sys.argv[5], validate=True))
record = Path(sys.argv[6])
record_sha = sys.argv[7]
info = record.lstat()
if (
    stat.S_ISLNK(info.st_mode)
    or not stat.S_ISREG(info.st_mode)
    or sha256_file(record) != record_sha
):
    raise RuntimeError("durable sync evidence is missing or mismatched")
target = current_pointer_target(root, publisher_uid, publisher_gid)
if (
    target != publication["pointer_target"]
    or Path(target).stem != publication["pointer_sha256"]
):
    raise RuntimeError("active source pointer changed before cleanup")
generation = Path(publication["generation_path"])
validate_frozen_ownership(generation, publisher_uid, publisher_gid)
if (
    hashlib.sha256((generation / "GENERATION.json").read_bytes()).hexdigest()
    != publication["generation_sha256"]
):
    raise RuntimeError("published generation is not frozen before cleanup")
remove_transaction(transaction)
print(f"LABCTL_SOURCE_CLEANED={transaction}")
print(f"LABCTL_SOURCE_CLEANED_BYTES={summary['bytes']}")
print(f"LABCTL_SOURCE_CLEANED_ENTRIES={summary['entries']}")
"""


FAILURE = SUPPORT + _CLEANUP + r"""
publisher_uid, publisher_gid = require_publisher(sys.argv[1], sys.argv[2])
requested_root = Path(sys.argv[3])
root = requested_root.resolve(strict=True)
if requested_root.is_symlink() or str(root) != str(requested_root):
    raise RuntimeError("source-generation root must already be canonical")
validate_controller_root(root, publisher_uid, publisher_gid)
transaction, summary = checked_transaction(
    root, sys.argv[4], publisher_uid, publisher_gid
)
error = base64.b64decode(sys.argv[5], validate=True).decode(errors="replace")[:2048]
failures = root / "failures"
owned_directory(failures, publisher_uid, publisher_gid)
manifests = list(failures.iterdir())
if len(manifests) >= 256:
    raise RuntimeError("failed-transaction manifest bound reached (256)")
document = {
    "cleanup_policy": {
        "max_transaction_bytes": 40 * 1024**3,
        "max_transaction_entries": 1_000_000,
        "min_free_bytes_before_next_transaction": 20 * 1024**3,
    },
    "error": error,
    "error_sha256": hashlib.sha256(error.encode()).hexdigest(),
    "schema_version": "stwo.gpu-lab.source-failure.v1",
    "transaction": str(transaction),
    "transaction_bytes": summary["bytes"],
    "transaction_entries": summary["entries"],
}
payload = canonical(document)
manifest = failures / f"{transaction.name}.json"
write_once(manifest, payload, publisher_uid, publisher_gid, mode=0o444)
validate_failure_manifest(
    manifest, transaction, summary, publisher_uid, publisher_gid
)
remove_transaction(transaction)
print(f"LABCTL_SOURCE_FAILURE={manifest}")
print(f"LABCTL_SOURCE_FAILURE_SHA256={hashlib.sha256(payload).hexdigest()}")
print(f"LABCTL_SOURCE_FAILURE_BYTES={summary['bytes']}")
print(f"LABCTL_SOURCE_FAILURE_ENTRIES={summary['entries']}")
"""


RECONCILE = SUPPORT + _CLEANUP + r"""
publisher_uid, publisher_gid = require_publisher(sys.argv[1], sys.argv[2])
requested_root = Path(sys.argv[3])
requested_root.mkdir(mode=0o755, parents=True, exist_ok=True)
root = requested_root.resolve()
if requested_root.is_symlink() or str(root) != str(requested_root):
    raise RuntimeError("source-generation root must already be canonical")
validate_controller_root(root, publisher_uid, publisher_gid)
transactions = root / "transactions"
failures = root / "failures"
owned_directory(transactions, publisher_uid, publisher_gid)
owned_directory(failures, publisher_uid, publisher_gid)
existing_failures = list(failures.iterdir())
transactions_to_reconcile = sorted(transactions.iterdir())
new_manifest_count = sum(
    not os.path.lexists(failures / f"{path.name}.json")
    for path in transactions_to_reconcile
)
if len(existing_failures) + new_manifest_count > 256:
    raise RuntimeError("failed-transaction manifest bound reached (256)")
for transaction_path in transactions_to_reconcile:
    transaction, summary = checked_transaction(
        root, str(transaction_path), publisher_uid, publisher_gid
    )
    document = {
        "cleanup_policy": {
            "max_transaction_bytes": 40 * 1024**3,
            "max_transaction_entries": 1_000_000,
            "min_free_bytes_before_next_transaction": 20 * 1024**3,
        },
        "error": "reconciled incomplete transaction after controller interruption",
        "schema_version": "stwo.gpu-lab.source-failure.v1",
        "transaction": str(transaction),
        "transaction_bytes": summary["bytes"],
        "transaction_entries": summary["entries"],
    }
    manifest = failures / f"{transaction.name}.json"
    manifest_existed = os.path.lexists(manifest)
    if not manifest_existed:
        write_once(
            manifest, canonical(document), publisher_uid, publisher_gid, mode=0o444
        )
    validate_failure_manifest(
        manifest, transaction, summary, publisher_uid, publisher_gid,
        allow_partial=manifest_existed,
    )
    remove_transaction(transaction)
print(f"LABCTL_SOURCE_RECONCILED={len(transactions_to_reconcile)}")
"""
