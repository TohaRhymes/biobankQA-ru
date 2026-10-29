"""Integrity and privacy checks for the reduced code-and-results release."""
import hashlib
import json
import re
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".json", ".jsonl", ".csv", ".toml", ".ini",
    ".yml", ".yaml", ".sh", ".feature", ".tex", ".bib",
}
SECRET = re.compile(
    rb"(?i)(?:sk-(?:proj-|live-|or-v1-)?[A-Za-z0-9_-]{20,}|"
    rb"AIza[0-9A-Za-z_-]{30,}|ya29\.[0-9A-Za-z_-]{20,}|"
    rb"Bearer\s+[A-Za-z0-9._-]{20,}|-----BEGIN[^\r\n]*PRIVATE KEY)"
)
EMAIL = re.compile(rb"(?i)[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}")
PRIVATE_LINK = re.compile(rb"https?://(?:docs|drive)\.google\.com/")
LOCAL_PATH = re.compile(rb"/" + rb"Users" + rb"/[^\s\"']+")


def _read_scan_bytes(path: Path):
    if path.suffix.lower() == ".docx":
        with zipfile.ZipFile(path) as archive:
            assert not any(n.startswith("word/comments") for n in archive.namelist()), path
            return [archive.read(n) for n in archive.namelist() if n.endswith((".xml", ".rels"))]
    if path.suffix.lower() in {".png", ".pdf", ".npy"}:
        return []
    return [path.read_bytes()]


def verify(root=ROOT):
    root = Path(root)
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    count = 0
    allowed = set()
    for entry in manifest["files"]:
        rel = Path(entry["path"])
        assert not rel.is_absolute() and ".." not in rel.parts, entry["path"]
        path = root / rel
        assert path.is_file(), entry["path"]
        content = path.read_bytes()
        assert len(content) == entry["bytes"], ("size mismatch", entry["path"])
        assert hashlib.sha256(content).hexdigest() == entry["sha256"], ("hash mismatch", entry["path"])
        allowed.add(rel.as_posix())
        count += 1
    allowed.add("MANIFEST.json")
    for path in root.rglob("*"):
        if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(root).as_posix()
        assert rel in allowed, ("unmanifested file", rel)
        assert path.suffix.lower() not in {".xlsx", ".xls", ".pem", ".key"}, rel
        for content in _read_scan_bytes(path):
            # This verifier necessarily contains credential-pattern syntax.
            if rel != "code/verify_release.py":
                assert not SECRET.search(content), ("possible credential", rel)
            assert not PRIVATE_LINK.search(content), ("private Drive link", rel)
            assert not LOCAL_PATH.search(content), ("local absolute path", rel)
            assert not EMAIL.search(content), ("email address", rel)
    for rel in ("data", "benchmark/outputs", "benchmark/rag/index", "manuscript"):
        assert not (root / rel).exists(), ("withheld artifacts", rel)
    forbidden = {".jsonl", ".xlsx", ".xls", ".docx", ".pdf", ".pkl", ".npy", ".zip"}
    for rel in allowed:
        assert Path(rel).suffix.lower() not in forbidden, ("withheld artifact format", rel)
    return count


if __name__ == "__main__":
    print("PASS release integrity/privacy checks:", verify(), "manifested files")
