# HAICAnchor — On-Chain Receipt Anchoring

Stores SHA3-256 Merkle roots produced by `utils/merkle.py` on an EVM chain.
Append-only, gas-cheap, indexer-friendly.

## Layout

| Path                          | Purpose                                          |
|-------------------------------|--------------------------------------------------|
| `contracts/HAICAnchor.sol`    | The anchor contract (Solidity 0.8.24)            |
| `test/HAICAnchor.t.sol`       | Foundry test suite (unit + fuzz)                 |
| `script/Deploy.s.sol`         | Sepolia/Holesky deploy script                    |
| `anchor_client.py`            | Python client (web3.py) — submit and verify      |
| `foundry.toml`                | Foundry config (RPC endpoints, etherscan)        |

## Hash compatibility

Solidity's `keccak256` is **not** SHA3-256 (NIST). All Merkle hashing happens
off-chain in `utils/merkle.py`. The contract only stores the resulting `bytes32`.
This is intentional and documented in the contract NatSpec.

## Local Foundry test

```bash
cd onchain
curl -L https://foundry.paradigm.xyz | bash
foundryup
forge install foundry-rs/forge-std --no-commit
forge test -vv
```

## Sepolia deploy

```bash
export DEPLOYER_PRIVATE_KEY=0x...           # funded with Sepolia ETH
export SEPOLIA_RPC_URL=https://sepolia.infura.io/v3/<key>
export ETHERSCAN_API_KEY=<key>              # optional, for --verify

forge script script/Deploy.s.sol --rpc-url sepolia --broadcast --verify
```

The deployed address is printed at the end. Save it as `HAIC_ANCHOR_ADDRESS`.

## Live anchor from Python

```bash
pip install web3
export RPC_URL=https://sepolia.infura.io/v3/<key>
export HAIC_ANCHOR_ADDRESS=0x...
export DEPLOYER_PRIVATE_KEY=0x...

# Anchor a Maestro local-receipt JSON:
python -m onchain.anchor_client \
    --receipt path/to/receipt.json \
    --kind session \
    --dry-run        # remove --dry-run to broadcast
```

## Verifying after the fact

```python
from onchain.anchor_client import AnchorClient
c = AnchorClient(rpc_url=..., contract_address=..., private_key=None)
print(c.verify("sess_42", merkle_root_hex))   # → True/False
print(c.get_anchor("sess_42"))                # → full record
```

## Gas

Anchoring one root costs ~70k gas (one `SSTORE` + one `LOG3`).
On Sepolia at 30 gwei this is ~$0.005 per anchor.
