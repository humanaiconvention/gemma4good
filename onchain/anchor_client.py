"""
anchor_client.py — Submit Merkle roots from utils/merkle.py to HAICAnchor.

The contract stores 32-byte values; SHA3-256 hex output from utils/merkle.py
is exactly 64 hex chars (32 bytes), so anchoring is a direct bytes-cast.

Usage:
    from onchain.anchor_client import AnchorClient
    client = AnchorClient(
        rpc_url="https://sepolia.infura.io/v3/...",
        contract_address="0x...",
        private_key=os.environ["DEPLOYER_PRIVATE_KEY"],
    )
    tx = client.anchor_session(session_id="sess_42", root_hex=root, kind="session", node_count=12)
    print(tx["hash"], tx["block"])
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from utils.merkle import sha3_256_hex

# web3 is an optional dependency for live anchoring.
try:
    from web3 import Web3
    from web3.middleware import ExtraDataToPOAMiddleware  # for Sepolia/Holesky
    _HAS_WEB3 = True
except ImportError:
    _HAS_WEB3 = False


_KIND_MAP = {"session": 0, "training": 1, "consent": 2}

# Minimal ABI — enough to call anchor() / verify() / getAnchor().
ANCHOR_ABI = [
    {
        "type": "function", "name": "anchor",
        "inputs": [
            {"name": "sessionId", "type": "bytes32"},
            {"name": "root",      "type": "bytes32"},
            {"name": "kind",      "type": "uint8"},
            {"name": "nodeCount", "type": "uint32"},
        ],
        "outputs": [], "stateMutability": "nonpayable",
    },
    {
        "type": "function", "name": "verify",
        "inputs": [
            {"name": "sessionId", "type": "bytes32"},
            {"name": "expected", "type": "bytes32"},
        ],
        "outputs": [{"name": "", "type": "bool"}],
        "stateMutability": "view",
    },
    {
        "type": "function", "name": "getAnchor",
        "inputs": [{"name": "sessionId", "type": "bytes32"}],
        "outputs": [{
            "components": [
                {"name": "root",        "type": "bytes32"},
                {"name": "blockNumber", "type": "uint64"},
                {"name": "timestamp",   "type": "uint64"},
                {"name": "submitter",   "type": "address"},
                {"name": "kind",        "type": "uint8"},
                {"name": "nodeCount",   "type": "uint32"},
            ],
            "name": "", "type": "tuple",
        }],
        "stateMutability": "view",
    },
]


def session_id_to_bytes32(session_id: str) -> bytes:
    """Hash a string sessionId to a deterministic bytes32 via SHA3-256."""
    return bytes.fromhex(sha3_256_hex(session_id))


def hex_to_bytes32(hex_str: str) -> bytes:
    """Validate and convert a 64-char hex root → 32-byte value."""
    h = hex_str.removeprefix("0x")
    if len(h) != 64:
        raise ValueError(f"expected 64 hex chars, got {len(h)}")
    return bytes.fromhex(h)


class AnchorClient:
    """Thin web3.py wrapper around HAICAnchor."""

    def __init__(
        self,
        rpc_url: str,
        contract_address: str,
        private_key: Optional[str] = None,
        chain_id: Optional[int] = None,
    ):
        if not _HAS_WEB3:
            raise RuntimeError("web3 not installed: pip install web3")
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        # Sepolia/Holesky use a longer extraData field (PoA).
        self.w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

        self.contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(contract_address),
            abi=ANCHOR_ABI,
        )
        self.private_key = private_key
        self.account = self.w3.eth.account.from_key(private_key) if private_key else None
        self.chain_id = chain_id or self.w3.eth.chain_id

    # ── writes ────────────────────────────────────────────────────────────────
    def anchor_session(
        self,
        session_id: str,
        root_hex: str,
        kind: str = "session",
        node_count: int = 0,
    ) -> dict:
        """Submit an anchor tx and wait for receipt."""
        if self.account is None:
            raise RuntimeError("private_key required for writes")
        kind_int = _KIND_MAP[kind.lower()]
        sid = session_id_to_bytes32(session_id)
        root = hex_to_bytes32(root_hex)

        nonce = self.w3.eth.get_transaction_count(self.account.address)
        tx = self.contract.functions.anchor(sid, root, kind_int, node_count).build_transaction({
            "from": self.account.address,
            "nonce": nonce,
            "chainId": self.chain_id,
        })
        signed = self.account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
        return {
            "hash": receipt.transactionHash.hex(),
            "block": receipt.blockNumber,
            "status": receipt.status,
            "gas_used": receipt.gasUsed,
        }

    # ── reads ─────────────────────────────────────────────────────────────────
    def verify(self, session_id: str, expected_root_hex: str) -> bool:
        sid = session_id_to_bytes32(session_id)
        return self.contract.functions.verify(sid, hex_to_bytes32(expected_root_hex)).call()

    def get_anchor(self, session_id: str) -> dict:
        sid = session_id_to_bytes32(session_id)
        a = self.contract.functions.getAnchor(sid).call()
        return {
            "root":         "0x" + a[0].hex(),
            "block_number": a[1],
            "timestamp":    a[2],
            "submitter":    a[3],
            "kind":         {0: "session", 1: "training", 2: "consent"}[a[4]],
            "node_count":   a[5],
        }


def cli() -> None:
    """Tiny CLI: generate an anchor from a Maestro-style local receipt JSON."""
    import argparse
    parser = argparse.ArgumentParser(description="Anchor a HAIC Merkle root on-chain.")
    parser.add_argument("--receipt", required=True, help="path to receipt JSON (must contain merkle_root + session_id)")
    parser.add_argument("--rpc", default=os.environ.get("RPC_URL"))
    parser.add_argument("--contract", default=os.environ.get("HAIC_ANCHOR_ADDRESS"))
    parser.add_argument("--key", default=os.environ.get("DEPLOYER_PRIVATE_KEY"))
    parser.add_argument("--kind", default="session", choices=list(_KIND_MAP.keys()))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    receipt = json.loads(Path(args.receipt).read_text())
    sid = receipt["session_id"]
    root = receipt["merkle_root"]
    nodes = int(receipt.get("node_count", 0))

    print(f"session_id   : {sid}")
    print(f"sid → bytes32: 0x{session_id_to_bytes32(sid).hex()}")
    print(f"merkle_root  : 0x{hex_to_bytes32(root).hex()}")
    print(f"kind         : {args.kind}")
    print(f"node_count   : {nodes}")
    if args.dry_run:
        print("(dry run — no tx sent)")
        return

    client = AnchorClient(args.rpc, args.contract, args.key)
    result = client.anchor_session(sid, root, args.kind, nodes)
    print("anchored:", json.dumps(result, indent=2))


if __name__ == "__main__":
    cli()
