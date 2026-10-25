"""Lossless scheduler-log storage after engine shutdown; outside measurement."""
import gzip
import hashlib
from pathlib import Path


def open_text(directory, name):
    path = Path(directory) / name
    if path.exists():
        return path.open('rt')
    return gzip.open(path.with_name(path.name + '.gz'), 'rt')


def compress_scheduler_log(directory):
    path = Path(directory) / 'scheduler_events.jsonl'
    if not path.exists():
        return {}
    compressed = path.with_name(path.name + '.gz')
    raw_hash, raw_bytes = hashlib.sha256(), 0
    with path.open('rb') as source, gzip.open(compressed, 'xb', compresslevel=1) as target:
        for chunk in iter(lambda: source.read(8 * 1024**2), b''):
            raw_hash.update(chunk)
            raw_bytes += len(chunk)
            target.write(chunk)
    verified, restored_bytes = hashlib.sha256(), 0
    with gzip.open(compressed, 'rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b''):
            verified.update(chunk)
            restored_bytes += len(chunk)
    if verified.digest() != raw_hash.digest() or restored_bytes != raw_bytes:
        raise RuntimeError('compressed scheduler log failed roundtrip; original retained')
    # Only this run's known log, after a verified lossless copy exists.
    path.unlink()
    return {path.name: {'file': compressed.name, 'raw_sha256': raw_hash.hexdigest(), 'raw_bytes': raw_bytes}}
