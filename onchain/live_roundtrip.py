"""
live_roundtrip.py — Deploy HAICAnchor to a local Anvil node and run a live
anchor + verify roundtrip using AnchorClient.

This is the free/local alternative to a Sepolia deployment.

Usage:
    python onchain/live_roundtrip.py

Requirements:
    - forge/anvil in PATH (installed via foundryup)
    - web3>=7: pip install web3
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
ONCHAIN = Path(__file__).resolve().parent

from utils.merkle import sha3_256_hex, merkle_root, hash_items_to_leaves

# Anvil's well-known pre-funded account #0 (10_000 ETH).
ANVIL_PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
ANVIL_RPC = "http://127.0.0.1:8545"
CHAIN_ID = 31337  # Anvil default

# Minimal HAICAnchor ABI — mirror of anchor_client.py ANCHOR_ABI.
from onchain.anchor_client import ANCHOR_ABI, AnchorClient, session_id_to_bytes32


def start_anvil() -> subprocess.Popen:
    """Launch anvil in the background and wait until RPC is responsive."""
    proc = subprocess.Popen(
        ["anvil", "--block-time", "1", "--port", "8545", "--silent"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    from web3 import Web3
    w3 = Web3(Web3.HTTPProvider(ANVIL_RPC))
    for _ in range(30):
        try:
            if w3.is_connected():
                print(f"  Anvil running (pid={proc.pid}, chain_id={w3.eth.chain_id})")
                return proc
        except Exception:
            pass
        time.sleep(0.5)
    proc.terminate()
    raise RuntimeError("Anvil failed to start in 15 s")


def compile_and_get_bytecode() -> tuple[str, list]:
    """Compile HAICAnchor.sol with forge build and return (bytecode, abi)."""
    out_dir = ONCHAIN / "out" / "HAICAnchor.sol"
    # Run forge build from the onchain directory.
    result = subprocess.run(
        ["forge", "build", "--silent"],
        cwd=str(ONCHAIN),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"forge build failed:\n{result.stderr}")

    artifact = out_dir / "HAICAnchor.json"
    data = json.loads(artifact.read_text())
    bytecode = data["bytecode"]["object"]
    abi = data["abi"]
    return bytecode, abi


def deploy_contract(w3, bytecode: str, abi: list, private_key: str) -> str:
    """Deploy HAICAnchor and return its address."""
    account = w3.eth.account.from_key(private_key)
    contract = w3.eth.contract(abi=abi, bytecode=bytecode)
    nonce = w3.eth.get_transaction_count(account.address)
    tx = contract.constructor().build_transaction({
        "from": account.address,
        "nonce": nonce,
        "chainId": CHAIN_ID,
    })
    signed = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=30)
    print(f"  Deployed at: {receipt.contractAddress} (block {receipt.blockNumber})")
    return receipt.contractAddress


def main() -> None:
    print("=" * 60)
    print("HAICAnchor LIVE ROUNDTRIP — Anvil local EVM")
    print("=" * 60)

    print("\n[1] Compiling HAICAnchor.sol …")
    bytecode, abi = compile_and_get_bytecode()
    print(f"  Bytecode size: {len(bytecode)//2} bytes")

    print("\n[2] Starting Anvil …")
    anvil = start_anvil()

    try:
        from web3 import Web3
        w3 = Web3(Web3.HTTPProvider(ANVIL_RPC))

        print("\n[3] Deploying HAICAnchor …")
        address = deploy_contract(w3, bytecode, abi, ANVIL_PRIVATE_KEY)

        print("\n[4] Building session receipt …")
        messages = [
            {"role": "user",      "content": "Tell me about your AI use."},
            {"role": "assistant", "content": "[PIVOT: SHADOW] When did that start?"},
            {"role": "user",      "content": "Last month when my boss installed it."},
        ]
        leaves = hash_items_to_leaves(messages)
        root_hex = merkle_root(leaves)
        session_id = "live_session_001"
        node_count = len(leaves)
        print(f"  session_id  : {session_id}")
        print(f"  merkle_root : 0x{root_hex}")
        print(f"  node_count  : {node_count}")

        print("\n[5] Anchoring via AnchorClient …")
        client = AnchorClient(
            rpc_url=ANVIL_RPC,
            contract_address=address,
            private_key=ANVIL_PRIVATE_KEY,
            chain_id=CHAIN_ID,
        )
        result = client.anchor_session(
            session_id=session_id,
            root_hex=root_hex,
            kind="session",
            node_count=node_count,
        )
        print(f"  tx hash     : {result['hash']}")
        print(f"  block       : {result['block']}")
        print(f"  status      : {'OK' if result['status'] == 1 else 'FAIL'}")
        print(f"  gas used    : {result['gas_used']}")

        print("\n[6] Verifying root on-chain …")
        ok = client.verify(session_id, root_hex)
        print(f"  verify(correct root)  : {ok}")
        assert ok, "verify(correct root) returned False"

        wrong_root = sha3_256_hex("wrong_payload")
        not_ok = client.verify(session_id, wrong_root)
        print(f"  verify(wrong root)    : {not_ok}")
        assert not not_ok, "verify(wrong root) returned True"

        print("\n[7] Fetching anchor record …")
        rec = client.get_anchor(session_id)
        print(json.dumps(rec, indent=2))
        assert rec["root"] == "0x" + root_hex
        assert rec["node_count"] == node_count
        assert rec["kind"] == "session"

        print("\n" + "=" * 60)
        print("LIVE ROUNDTRIP PASSED ✓")
        print("=" * 60)

        # Save result for CI / commit artifact.
        out = ROOT / "onchain" / "live_roundtrip_result.json"
        out.write_text(json.dumps({
            "status": "PASS",
            "contract_address": address,
            "session_id": session_id,
            "merkle_root": "0x" + root_hex,
            "tx_hash": result["hash"],
            "block": result["block"],
            "gas_used": result["gas_used"],
            "anchor_record": rec,
        }, indent=2))
        print(f"Result saved → {out}")

    finally:
        anvil.terminate()
        anvil.wait()
        print("\nAnvil stopped.")


if __name__ == "__main__":
    main()
