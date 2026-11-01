import json
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("jsonlines")

from adalflow.utils.file_io import append_to_jsonl


def test_append_to_jsonl_is_thread_safe(tmp_path):
    """Concurrent appends must each produce one complete json line."""
    path = str(tmp_path / "calls.jsonl")
    n = 200
    payload = "x" * 20000

    def write(i):
        append_to_jsonl(path, {"id": i, "text": payload})

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write, range(n)))

    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    assert sorted(r["id"] for r in records) == list(range(n))
