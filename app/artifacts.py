import hashlib
import json
import os
from pathlib import Path
import tempfile


# # Compute the SHA256 hash of a file to check integrity
# def SHA256(filePath: Path):
#     digest = hashlib.sha256()
#     with Path(filePath).open('rb') as f:
#         for chunk in iter(lambda: f.read(1024 * 1024), b""):
#             digest.update(chunk)
#     return digest.hexdigest()


# # Identify a saved checkpoint
# def checkpoint_id(directory: Path):
#     directory = Path(directory)
#     files = sorted(p for p in directory.rglob('*') if p.is_file())
#     if not files:
#         raise ValueError("Empty checkpoint directory")
#     entries = [(p.relative_to(directory).as_posix(), SHA256(p)) for p in files]
#     return hashlib.sha256(json.dumps(entries).encode('utf-8')).hexdigest()


# write content to a file, optionally replacing it if it already exists
def write_bytes(path, content, replace=False):
    """Save a complete file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        if path.read_bytes() != content:
            raise ValueError(f"Refusing to overwrite different contents: {path}")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
        except OSError:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)



# save json in clean, readable format
def write_json(path, value, replace=False):
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    write_bytes(path, content.encode("utf-8"), replace=replace)
