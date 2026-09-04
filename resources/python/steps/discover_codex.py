#!/usr/bin/env python3
"""@02 discover_codex — find Codex session files."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import discovery_dict, emit  # noqa: E402
from gh.discover import discover_harness  # noqa: E402


def main(argv: list[str]) -> int:
    days = int(argv[1]) if len(argv) > 1 else 14
    out_path = Path(argv[2]) if len(argv) > 2 else None
    result = discover_harness("codex", days)
    payload = discovery_dict(result)
    text = json.dumps(payload)
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    emit(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
