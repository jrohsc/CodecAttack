"""Validate explicit ensemble carrier identities and hashes."""
import json
import hashlib
from pathlib import Path

def validate_cohort(path):
    cohort = json.loads(Path(path).read_text())
    rows = cohort['pairs']
    if not rows or len({r['example_id'] for r in rows}) != len(rows):
        raise ValueError('Empty or duplicate cohort')
    for r in rows:
        if not r['target'].strip() or not r['prompt'].strip():
            raise ValueError('Empty target or prompt')
        p = Path(r['carrier_path'])
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != r['carrier_sha256']:
            raise ValueError(f'Missing or changed carrier: {p}')
    return cohort
