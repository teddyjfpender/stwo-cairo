"""Remote helpers for frozen source trees and bounded transaction cleanup."""

SUPPORT = r"""
def git_index_inventory(repo):
    dot_git = repo / ".git"
    if dot_git.is_dir():
        index = dot_git / "index"
    else:
        prefix, location = dot_git.read_text().strip().split(": ", 1)
        if prefix != "gitdir":
            raise RuntimeError("invalid Git worktree control path")
        index = (dot_git.parent / location).resolve() / "index"
    info = index.lstat()
    digest = hashlib.sha256()
    with open(index, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return (
        info.st_ino, info.st_mode, info.st_uid, info.st_gid, info.st_nlink,
        info.st_size, info.st_mtime_ns, digest.digest(),
    )

def readonly_git(repo, *args):
    before = git_index_inventory(repo)
    result = git(repo, *args)
    if git_index_inventory(repo) != before:
        raise RuntimeError(f"read-only Git command mutated index: {args}")
    return result

def index_clean(repo):
    index = []
    for raw in readonly_git(repo, "ls-files", "--stage", "-z").stdout.split("\0"):
        if not raw:
            continue
        metadata, path = raw.split("\t", 1)
        mode, object_id, stage = metadata.split(" ")
        if stage != "0":
            return False
        index.append((mode, object_id, path))
    tree = []
    for raw in readonly_git(
        repo, "ls-tree", "-r", "-z", "--full-tree", "HEAD"
    ).stdout.split("\0"):
        if not raw:
            continue
        metadata, path = raw.split("\t", 1)
        mode, _, object_id = metadata.split(" ")
        tree.append((mode, object_id, path))
    return index == tree

def nodes(root):
    directories, regular, links, special = [root], [], [], []
    pending = [root]
    while pending:
        directory = pending.pop()
        for entry in os.scandir(directory):
            path = Path(entry.path)
            info = path.lstat()
            if stat.S_ISDIR(info.st_mode):
                directories.append(path)
                pending.append(path)
            elif stat.S_ISREG(info.st_mode):
                regular.append(path)
            elif stat.S_ISLNK(info.st_mode):
                links.append(path)
            else:
                special.append(path)
    return directories, regular, links, special

def tracked_symlinks(repo, expected):
    output = readonly_git(repo, "ls-files", "--stage", "-z").stdout
    links = set()
    for raw in output.split("\0"):
        if not raw:
            continue
        metadata, path = raw.split("\t", 1)
        if metadata.split(" ", 1)[0] == "120000":
            links.add(path)
    for item in expected["entries"]:
        links.discard(item["path"])
        if item["kind"] == "symlink":
            links.add(item["path"])
    return links

def expected_generation_links(generation, identities):
    expected = set()
    for name, document in identities.items():
        repo = generation / "repos" / name
        for rel in tracked_symlinks(repo, document):
            expected.add((repo / rel).relative_to(generation).as_posix())
    return expected

def require_publisher_owner(path, info, publisher_uid, publisher_gid):
    if info.st_uid != publisher_uid or info.st_gid != publisher_gid:
        raise RuntimeError(f"controller object has the wrong publisher owner: {path}")

def validate_frozen_ownership(
    generation, publisher_uid, publisher_gid, root_mode=0o555
):
    info = generation.lstat()
    require_publisher_owner(generation, info, publisher_uid, publisher_gid)
    if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != root_mode:
        raise RuntimeError("published generation root is mutable or invalid")
    directories, regular, links, special = nodes(generation)
    if special:
        raise RuntimeError(f"published generation contains special objects: {special[:3]}")
    for directory in directories:
        info = directory.lstat()
        require_publisher_owner(directory, info, publisher_uid, publisher_gid)
        expected_mode = root_mode if directory == generation else 0o555
        if stat.S_IMODE(info.st_mode) != expected_mode:
            raise RuntimeError(f"published directory is mutable: {directory}")
    for path in regular:
        info = path.lstat()
        require_publisher_owner(path, info, publisher_uid, publisher_gid)
        if info.st_nlink != 1 or stat.S_IMODE(info.st_mode) not in (0o444, 0o555):
            raise RuntimeError(f"published file is mutable: {path}")
    for path in links:
        require_publisher_owner(path, path.lstat(), publisher_uid, publisher_gid)
    return directories, regular, links

def validate_generation(
    generation, identities, generation_payload, publisher_uid, publisher_gid,
    root_mode=0o555,
):
    _, _, links = validate_frozen_ownership(
        generation, publisher_uid, publisher_gid, root_mode
    )
    if {entry.name for entry in os.scandir(generation)} != {"GENERATION.json", "repos"}:
        raise RuntimeError("published generation has unexpected top-level objects")
    manifest = generation / "GENERATION.json"
    manifest_info = manifest.lstat()
    if (
        not stat.S_ISREG(manifest_info.st_mode)
        or stat.S_IMODE(manifest_info.st_mode) != 0o444
        or manifest.read_bytes() != generation_payload
    ):
        raise RuntimeError("published GENERATION.json is mutable or invalid")
    repos = generation / "repos"
    repos_info = repos.lstat()
    if (
        not stat.S_ISDIR(repos_info.st_mode)
        or repos_info.st_uid != publisher_uid
        or repos_info.st_gid != publisher_gid
    ):
        raise RuntimeError("published repos object is invalid")
    if {entry.name for entry in os.scandir(repos)} != set(identities):
        raise RuntimeError("published repo set mismatch")
    expected_links = expected_generation_links(generation, identities)
    validate_frozen_ownership(generation, publisher_uid, publisher_gid, root_mode)
    actual_links = {path.relative_to(generation).as_posix() for path in links}
    if actual_links != expected_links:
        raise RuntimeError("published generation has unexpected symlink descendants")
    for path in links:
        target = os.readlink(path)
        if os.path.isabs(target):
            raise RuntimeError(f"published symlink is absolute: {path}")
        repo = generation / "repos" / path.relative_to(generation / "repos").parts[0]
        resolved = (path.parent / target).resolve(strict=True)
        try:
            resolved.relative_to(repo.resolve(strict=True))
        except ValueError:
            raise RuntimeError(f"published symlink escapes its repo: {path}")
        relative_target = resolved.relative_to(repo.resolve(strict=True)).as_posix()
        if relative_target == ".git" or relative_target.startswith(".git/"):
            raise RuntimeError(f"published symlink targets Git control data: {path}")
    for name, expected in identities.items():
        repo = repos / name
        repo_info = repo.lstat()
        if not stat.S_ISDIR(repo_info.st_mode) or stat.S_IMODE(repo_info.st_mode) != 0o555:
            raise RuntimeError(f"published repo object is mutable or invalid: {name}")
        actual_identity = identity(repo)
        validate_frozen_ownership(generation, publisher_uid, publisher_gid, root_mode)
        clean_index = index_clean(repo)
        validate_frozen_ownership(generation, publisher_uid, publisher_gid, root_mode)
        if actual_identity != expected or not clean_index:
            raise RuntimeError(f"published repo identity mismatch: {name}")
    validate_frozen_ownership(generation, publisher_uid, publisher_gid, root_mode)

def freeze_generation(
    generation, identities, generation_payload, publisher_uid, publisher_gid
):
    if (generation / "GENERATION.json").read_bytes() != generation_payload:
        raise RuntimeError("generation manifest changed before freeze")
    expected_links = expected_generation_links(generation, identities)
    directories, regular, links, special = nodes(generation)
    if special:
        raise RuntimeError(f"generation contains special objects: {special[:3]}")
    for path in directories + regular + links:
        require_publisher_owner(path, path.lstat(), publisher_uid, publisher_gid)
    actual_links = {path.relative_to(generation).as_posix() for path in links}
    if actual_links != expected_links:
        raise RuntimeError("generation contains unexpected symlink descendants")
    for path in regular:
        info = path.lstat()
        if info.st_nlink != 1:
            raise RuntimeError(f"generation contains a hardlinked file: {path}")
        mode = info.st_mode
        os.chmod(path, 0o555 if mode & 0o111 else 0o444)
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        if directory != generation:
            os.chmod(directory, 0o555)
    os.chmod(generation, 0o555)
    validate_generation(
        generation, identities, generation_payload, publisher_uid, publisher_gid
    )

def inventory(root):
    directories, regular, links, special = nodes(root)
    entries = len(directories) + len(regular) + len(links) + len(special)
    size = sum(path.lstat().st_size for path in regular + links + special)
    hardlinks = sum(path.lstat().st_nlink != 1 for path in regular)
    return {
        "bytes": size, "entries": entries, "hardlinks": hardlinks,
        "special": len(special),
    }
"""
