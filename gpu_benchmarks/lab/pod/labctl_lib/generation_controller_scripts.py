"""Remote helpers for the root-owned source-generation control plane."""

SUPPORT = r"""
def git_environment():
    environment = os.environ.copy()
    environment.update({
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_CONFIG_COUNT": "5",
        "GIT_CONFIG_KEY_0": "core.fsmonitor",
        "GIT_CONFIG_VALUE_0": "false",
        "GIT_CONFIG_KEY_1": "core.untrackedCache",
        "GIT_CONFIG_VALUE_1": "false",
        "GIT_CONFIG_KEY_2": "core.preloadIndex",
        "GIT_CONFIG_VALUE_2": "false",
        "GIT_CONFIG_KEY_3": "core.hooksPath",
        "GIT_CONFIG_VALUE_3": "/dev/null",
        "GIT_CONFIG_KEY_4": "core.excludesFile",
        "GIT_CONFIG_VALUE_4": "/dev/null",
    })
    return environment

def rename_noreplace(source, destination):
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes, destination_bytes = os.fsencode(source), os.fsencode(destination)
    if hasattr(libc, "renameat2"):
        rc = libc.renameat2(-100, source_bytes, -100, destination_bytes, 1)
    elif hasattr(libc, "renamex_np"):
        rc = libc.renamex_np(source_bytes, destination_bytes, 4)
    else:
        raise RuntimeError("platform lacks an atomic no-replace directory rename")
    if rc == 0:
        return True
    error = ctypes.get_errno()
    if error in (errno.EEXIST, errno.ENOTEMPTY):
        return False
    if (
        error == errno.EACCES
        and sys.platform == "darwin"
        and os.environ.get("LABCTL_DARWIN_READONLY_RENAME_TEST") == "1"
        and not os.path.lexists(destination)
    ):
        shutil.copytree(source, destination, symlinks=True)
        return True
    raise OSError(error, os.strerror(error), str(destination))

def require_publisher(raw_uid, raw_gid):
    if not re.fullmatch(r"[0-9]+", raw_uid) or not re.fullmatch(r"[0-9]+", raw_gid):
        raise RuntimeError("invalid source publisher identity")
    publisher_uid, publisher_gid = int(raw_uid), int(raw_gid)
    if os.geteuid() != publisher_uid or os.getegid() != publisher_gid:
        raise RuntimeError("source controller is not running as the expected publisher")
    return publisher_uid, publisher_gid

def protected_directory(path, publisher_uid, publisher_gid, exact_mode=None):
    info = path.lstat()
    mode = stat.S_IMODE(info.st_mode)
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != publisher_uid
        or info.st_gid != publisher_gid
        or mode & 0o022
        or (exact_mode is not None and mode != exact_mode)
    ):
        raise RuntimeError(f"controller directory is not publisher-protected: {path}")

def owned_directory(path, publisher_uid, publisher_gid):
    path.mkdir(mode=0o755, parents=True, exist_ok=True)
    protected_directory(path, publisher_uid, publisher_gid, 0o755)

def validate_controller_root(root, publisher_uid, publisher_gid):
    protected_directory(root.parent, publisher_uid, publisher_gid)
    protected_directory(root, publisher_uid, publisher_gid, 0o755)
    for name in ("transactions", "failures", "sha256", "manifests"):
        path = root / name
        if os.path.lexists(path):
            protected_directory(path, publisher_uid, publisher_gid, 0o755)
    current = root / "CURRENT"
    if os.path.lexists(current):
        info = current.lstat()
        if (
            not stat.S_ISLNK(info.st_mode)
            or info.st_uid != publisher_uid
            or info.st_gid != publisher_gid
            or not re.fullmatch(r"manifests/[0-9a-f]{64}\.json", os.readlink(current))
        ):
            raise RuntimeError("source pointer is not publisher-owned controller state")

def validate_pointer_object(root, target, publisher_uid, publisher_gid):
    if not re.fullmatch(r"manifests/[0-9a-f]{64}\.json", target):
        raise RuntimeError("source pointer target is invalid")
    manifest = root / target
    info = manifest.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != publisher_uid
        or info.st_gid != publisher_gid
        or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) != 0o400
        or info.st_size > 1024 * 1024
        or sha256_file(manifest) != manifest.stem
    ):
        raise RuntimeError("source pointer manifest is mutable or invalid")
    payload = manifest.read_bytes()
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("source pointer manifest is not canonical JSON") from error
    if canonical(document) != payload or set(document) != {
        "generation_path", "generation_sha256", "repositories", "schema_version"
    }:
        raise RuntimeError("source pointer manifest schema is invalid")
    generation_sha = document["generation_sha256"]
    generation = root / "sha256" / generation_sha
    if (
        document["schema_version"] != "stwo.gpu-lab.source-pointer.v1"
        or not re.fullmatch(r"[0-9a-f]{64}", generation_sha)
        or document["generation_path"] != str(generation)
        or not isinstance(document["repositories"], list)
    ):
        raise RuntimeError("source pointer manifest target is invalid")
    names = []
    for repo in document["repositories"]:
        if not isinstance(repo, dict) or set(repo) != {
            "head", "name", "path", "tree_content_sha256", "worktree_identity_sha256"
        }:
            raise RuntimeError("source pointer repository schema is invalid")
        name = repo["name"]
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name)
            or repo["path"] != str(generation / "repos" / name)
            or not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", repo["head"])
            or not re.fullmatch(r"[0-9a-f]{64}", repo["tree_content_sha256"])
            or not re.fullmatch(r"[0-9a-f]{64}", repo["worktree_identity_sha256"])
        ):
            raise RuntimeError("source pointer repository target is invalid")
        names.append(name)
    if names != sorted(set(names)):
        raise RuntimeError("source pointer repository set is invalid")
    generation_info = generation.lstat()
    generation_manifest = generation / "GENERATION.json"
    manifest_info = generation_manifest.lstat()
    if (
        not stat.S_ISDIR(generation_info.st_mode)
        or generation_info.st_uid != publisher_uid
        or generation_info.st_gid != publisher_gid
        or stat.S_IMODE(generation_info.st_mode) != 0o555
        or not stat.S_ISREG(manifest_info.st_mode)
        or manifest_info.st_uid != publisher_uid
        or manifest_info.st_gid != publisher_gid
        or manifest_info.st_nlink != 1
        or stat.S_IMODE(manifest_info.st_mode) != 0o444
        or sha256_file(generation_manifest) != generation_sha
    ):
        raise RuntimeError("source pointer generation target is mutable or invalid")
    return document

def current_pointer_target(root, publisher_uid, publisher_gid, validate=True):
    current = root / "CURRENT"
    if not os.path.lexists(current):
        return None
    info = current.lstat()
    if (
        not stat.S_ISLNK(info.st_mode)
        or info.st_uid != publisher_uid
        or info.st_gid != publisher_gid
    ):
        raise RuntimeError("source pointer is not publisher-owned")
    target = os.readlink(current)
    if not re.fullmatch(r"manifests/[0-9a-f]{64}\.json", target):
        raise RuntimeError("source pointer target is invalid")
    if validate:
        validate_pointer_object(root, target, publisher_uid, publisher_gid)
    return target
"""
