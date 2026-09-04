#!/usr/bin/env python3
"""@00 selfcheck — bundled fixtures through the real analyzer."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import emit  # noqa: E402
from step_io import write_artifact  # noqa: E402
from gh.selfcheck import (  # noqa: E402
    failed_selfcheck,
    result_to_dict,
    run_selfcheck,
)


def main(argv: list[str]) -> int:
    # argv: selfcheck.py [out_path]
    out_path = argv[1] if len(argv) > 1 else None
    try:
        result = run_selfcheck()
    except Exception as exc:  # noqa: BLE001 — DAG must continue
        result = failed_selfcheck(f"raised {type(exc).__name__}: {exc}")
    payload = result_to_dict(result)
    write_artifact(out_path, payload)
    emit(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
