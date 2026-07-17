"""Resolve the active root-owned source generation for development use."""

from __future__ import annotations

import json
import re

from . import common as c
from . import generation
from . import generation_scripts
from . import runtime


SCHEMA = "stwo.gpu-lab.source-resolution.v1"
REPOSITORIES = ("stwo", "stwo-cairo")

REMOTE_RESOLVE = generation_scripts.SUPPORT + r"""
publisher_uid, publisher_gid = require_publisher(sys.argv[1], sys.argv[2])
requested_root = Path(sys.argv[3])
if not os.path.lexists(requested_root):
    raise RuntimeError("no active source generation; run `labctl sync` first")
root = requested_root.resolve(strict=True)
if requested_root.is_symlink() or str(root) != str(requested_root):
    raise RuntimeError("source-generation root must already be canonical")
validate_controller_root(root, publisher_uid, publisher_gid)
target = current_pointer_target(root, publisher_uid, publisher_gid)
if target is None:
    raise RuntimeError("no active source generation; run `labctl sync` first")
pointer = root / target
pointer_payload = pointer.read_bytes()
pointer_sha = hashlib.sha256(pointer_payload).hexdigest()
if pointer_sha != pointer.stem:
    raise RuntimeError("active source-pointer manifest digest mismatch")
pointer_document = validate_pointer_object(
    root, target, publisher_uid, publisher_gid
)
generation = Path(pointer_document["generation_path"])
generation_manifest = generation / "GENERATION.json"
generation_payload = generation_manifest.read_bytes()
try:
    generation_document = json.loads(generation_payload)
except (UnicodeDecodeError, json.JSONDecodeError) as error:
    raise RuntimeError("active generation manifest is not canonical JSON") from error
if (
    canonical(generation_document) != generation_payload
    or set(generation_document) != {"repositories", "schema_version"}
    or generation_document["schema_version"]
        != "stwo.gpu-lab.source-generation.v1"
    or not isinstance(generation_document["repositories"], list)
):
    raise RuntimeError("active generation manifest schema is invalid")

generation_repositories = {}
for repo in generation_document["repositories"]:
    if not isinstance(repo, dict) or set(repo) != {
        "head", "name", "relative_path", "tree_content_sha256",
        "worktree_identity_sha256",
    }:
        raise RuntimeError("active generation repository schema is invalid")
    name = repo["name"]
    if (
        not isinstance(name, str)
        or repo["relative_path"] != f"repos/{name}"
        or not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", repo["head"])
        or not re.fullmatch(r"[0-9a-f]{64}", repo["tree_content_sha256"])
        or not re.fullmatch(r"[0-9a-f]{64}", repo["worktree_identity_sha256"])
        or name in generation_repositories
    ):
        raise RuntimeError("active generation repository identity is invalid")
    generation_repositories[name] = repo

pointer_repositories = {
    repo["name"]: repo for repo in pointer_document["repositories"]
}
required = {"stwo", "stwo-cairo"}
if set(generation_repositories) != required or set(pointer_repositories) != required:
    raise RuntimeError("active generation must contain exactly stwo and stwo-cairo")
protected_directory(generation / "repos", publisher_uid, publisher_gid, 0o555)
resolved = {}
for name in sorted(required):
    source = generation / "repos" / name
    protected_directory(source, publisher_uid, publisher_gid, 0o555)
    expected = generation_repositories[name]
    pointer_repo = pointer_repositories[name]
    if pointer_repo != {
        "head": expected["head"],
        "name": name,
        "path": str(source),
        "tree_content_sha256": expected["tree_content_sha256"],
        "worktree_identity_sha256": expected["worktree_identity_sha256"],
    }:
        raise RuntimeError("source pointer and generation identities disagree")
    resolved[name] = {
        "head": expected["head"],
        "path": str(source),
        "tree_content_sha256": expected["tree_content_sha256"],
        "worktree_identity_sha256": expected["worktree_identity_sha256"],
    }
if current_pointer_target(root, publisher_uid, publisher_gid) != target:
    raise RuntimeError("active source pointer changed while resolving")
document = {
    "generation_path": str(generation),
    "generation_sha256": pointer_document["generation_sha256"],
    "pointer_path": str(root / "CURRENT"),
    "pointer_sha256": pointer_sha,
    "pointer_target": target,
    "repositories": resolved,
    "schema_version": "stwo.gpu-lab.source-resolution.v1",
}
print(canonical(document).decode(), end="")
"""


def _canonical(document: dict) -> str:
    return json.dumps(
        document, allow_nan=False, sort_keys=True, separators=(",", ":")
    )


def _parse(output: str, root: str = generation.SOURCE_ROOT) -> dict:
    try:
        document = json.loads(output)
    except json.JSONDecodeError as error:
        raise RuntimeError("invalid source-resolution receipt") from error
    if not isinstance(document, dict) or _canonical(document) != output:
        raise RuntimeError("source-resolution receipt is not canonical JSON")
    if set(document) != {
        "generation_path",
        "generation_sha256",
        "pointer_path",
        "pointer_sha256",
        "pointer_target",
        "repositories",
        "schema_version",
    }:
        raise RuntimeError("source-resolution receipt schema is invalid")
    generation_sha = document["generation_sha256"]
    pointer_sha = document["pointer_sha256"]
    if (
        document["schema_version"] != SCHEMA
        or not isinstance(generation_sha, str)
        or not isinstance(pointer_sha, str)
        or not re.fullmatch(r"[0-9a-f]{64}", generation_sha)
        or not re.fullmatch(r"[0-9a-f]{64}", pointer_sha)
        or document["generation_path"] != f"{root}/sha256/{generation_sha}"
        or document["pointer_path"] != f"{root}/CURRENT"
        or document["pointer_target"] != f"manifests/{pointer_sha}.json"
        or not isinstance(document["repositories"], dict)
        or tuple(sorted(document["repositories"])) != REPOSITORIES
    ):
        raise RuntimeError("source-resolution receipt target is invalid")
    for name in REPOSITORIES:
        repo = document["repositories"][name]
        if (
            not isinstance(repo, dict)
            or set(repo)
            != {
                "head",
                "path",
                "tree_content_sha256",
                "worktree_identity_sha256",
            }
            or repo["path"] != f"{document['generation_path']}/repos/{name}"
            or not all(
                isinstance(repo[key], str)
                for key in (
                    "head",
                    "path",
                    "tree_content_sha256",
                    "worktree_identity_sha256",
                )
            )
            or not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", repo["head"])
            or not re.fullmatch(r"[0-9a-f]{64}", repo["tree_content_sha256"])
            or not re.fullmatch(
                r"[0-9a-f]{64}", repo["worktree_identity_sha256"]
            )
        ):
            raise RuntimeError("source-resolution repository identity is invalid")
    return document


def active_source(ep: c.Endpoint, root: str = generation.SOURCE_ROOT) -> dict:
    if ep.user != "root":
        raise RuntimeError("source resolution requires the root controller endpoint")
    rc, output = generation._capture(ep, REMOTE_RESOLVE, [root], 30)
    if rc:
        raise RuntimeError(f"active source resolution failed: {output}")
    return _parse(output, root)


def cmd_resolve(_args) -> int:
    with c._lease_lock():
        _, _, ep = runtime._active()
    print(json.dumps(active_source(ep), allow_nan=False, indent=2, sort_keys=True))
    return 0
