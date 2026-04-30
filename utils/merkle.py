"""
merkle.py — Shared Merkle tree utilities for the HAIC framework.

All receipt hashing in the framework (session receipts, training receipts,
consent hashes) uses SHA3-256 for EVM/on-chain compatibility.

Previously this logic was duplicated in:
  - tools/haic_tools.py      (generate_receipt local fallback)
  - tools/incremental_grounding.py  (_sha256, _merkle_root)
  - maestro_integration/maestro_client.py  (_local_receipt)

Now all three import from here.
"""

import hashlib
import json
from typing import Union


def sha3_256_hex(data: Union[str, bytes]) -> str:
    """SHA3-256 hex digest of a string or bytes value."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha3_256(data).hexdigest()


def merkle_root(leaves: list[str]) -> str:
    """
    Compute a Merkle root from a list of hex-string leaves.

    Uses pairwise SHA3-256 reduction.  If the number of leaves is odd,
    the last leaf is duplicated before pairing.  An empty list hashes
    the literal string "empty".

    Args:
        leaves: list of hex-encoded hash strings

    Returns:
        64-character lowercase hex string — the Merkle root.
    """
    if not leaves:
        return sha3_256_hex("empty")

    nodes = list(leaves)
    while len(nodes) > 1:
        if len(nodes) % 2 == 1:
            nodes.append(nodes[-1])
        nodes = [
            sha3_256_hex(nodes[i] + nodes[i + 1])
            for i in range(0, len(nodes), 2)
        ]
    return nodes[0]


def hash_items_to_leaves(items: list[dict]) -> list[str]:
    """
    JSON-serialize each item with sorted keys and return SHA3-256 leaves.

    Useful for hashing message lists or consent dicts into Merkle leaves.

    Args:
        items: list of JSON-serializable dicts

    Returns:
        list of 64-character hex strings, one per item.
    """
    return [
        sha3_256_hex(json.dumps(item, sort_keys=True))
        for item in items
    ]
