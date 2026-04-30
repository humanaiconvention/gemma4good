"""
maestro_gateway/smoke_test.py — End-to-end live smoke against a deployed gateway.

Usage:
    MAESTRO_GATEWAY_BASE=https://your-deploy.up.railway.app \
        python -m maestro_gateway.smoke_test

Exits 0 on full success. Prints each step + receipt comparison.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maestro_integration.maestro_client import MaestroClient
from utils.merkle import sha3_256_hex, merkle_root, hash_items_to_leaves


def main() -> int:
    base = os.environ.get("MAESTRO_GATEWAY_BASE", "http://localhost:8000")
    print(f"== smoke against {base}")
    client = MaestroClient(base_url=base)

    # 1. health
    h = client.health()
    print(f"  health    : {h}")
    if h.get("status") != "ok":
        print("FAIL: health not ok"); return 1

    # 2. dev token
    tok = client.dev_token()
    print(f"  dev_token : {(tok or '')[:24]}...")
    if not tok:
        print("FAIL: no dev token"); return 1

    # 3. consent
    sid = f"smoke_{int(time.time())}"
    consent = {"transcript": "granted", "lattice": "granted"}
    r = client.submit_consent(sid, consent)
    print(f"  consent   : {r.get('consent_hash', '')[:16]}...")
    if "error" in r:
        print("FAIL:", r); return 1

    # 4. receipt — and verify it matches the local computation
    msgs = [{"role": "user", "content": f"smoke {i}"} for i in range(3)]
    receipt = client.submit_receipt(sid, msgs, consent)
    print(f"  receipt   : root={receipt['merkle_root'][:16]}... source={receipt.get('source')}")

    # Recompute locally and compare:
    local_leaves = hash_items_to_leaves(msgs)
    local_leaves.append(sha3_256_hex(json.dumps(consent, sort_keys=True)))
    expected = merkle_root(local_leaves)
    if receipt["merkle_root"] != expected:
        print(f"FAIL: gateway root != local root")
        print(f"  gateway:  {receipt['merkle_root']}")
        print(f"  expected: {expected}")
        return 1
    print("  ✓ gateway and local Merkle roots match — receipts are verifiable")

    # 5. echo chat
    out = client.chat([{"role": "user", "content": "ping"}])
    print(f"  chat      : {out and out.get('choices', [{}])[0].get('message', {}).get('content')!r}")

    print("== smoke OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
