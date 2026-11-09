#!/usr/bin/env python
"""Regenerate ``prompts.default.yaml`` from the built-in prompt bank.

    python scripts/export_prompts.py                 # ml bank -> prompts.default.yaml
    python scripts/export_prompts.py --domain hep_ph --out hep_prompts.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from researchclaw.prompts.export import write_default_prompts  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", default="ml", help="Prompt bank (ml, hep_ph, biology_metabolic)")
    parser.add_argument("--out", default="prompts.default.yaml", help="Output YAML path")
    args = parser.parse_args()

    path = write_default_prompts(args.out, domain=args.domain)
    print(f"wrote {path} from the {args.domain} bank")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
