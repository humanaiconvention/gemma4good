# Tier 3 Runbook — Live Deployments

This document describes how to take the Tier 3 components from this repo to a
live deployment. Both pieces (Maestro gateway + on-chain anchor) work
independently; you can deploy them in any order.

## 1 — Maestro Gateway

### 1a. Local Docker

```bash
cd maestro_gateway
docker compose up --build
# In another shell, from the repo root (smoke_test imports maestro_integration,
# which is not installed inside the container — run on the host):
cd ..   # back to D:/gemma4good
pip install -e ".[runtime]"   # one-time: ensures `requests` is available
MAESTRO_GATEWAY_BASE=http://localhost:8000 \
  python -m maestro_gateway.smoke_test
```

`smoke_test.py` checks `/health`, fetches a dev token, submits a consent record,
posts a receipt, and confirms the gateway-computed Merkle root matches the
locally-computed one (cross-language Merkle parity guarantee).

### 1b. Railway

```bash
# Login + project bootstrap (one time)
railway login
railway init
railway link

# Deploy
railway up

# Required env vars
railway variables --set MAESTRO_LAUNCH_MODE=test
railway variables --set MAESTRO_DEV_TOKEN=$(openssl rand -hex 16)
railway variables --set MAESTRO_RATE_LIMIT_S=0.5

# Verify
MAESTRO_GATEWAY_BASE=https://<your-app>.up.railway.app \
  python -m maestro_gateway.smoke_test
```

### 1c. Fly.io

```bash
cd maestro_gateway
flyctl launch --no-deploy --copy-config
flyctl secrets set MAESTRO_DEV_TOKEN=$(openssl rand -hex 16)
flyctl deploy
flyctl status
```

### 1d. Wiring the gateway into the notebook / CLI

The existing `MaestroClient` and `haic_tools.generate_receipt()` already pick up
`MAESTRO_GATEWAY_BASE` from the environment. Once your gateway is live:

```bash
export MAESTRO_GATEWAY_BASE=https://<your-app>.up.railway.app
python -c "from maestro_integration.maestro_client import MaestroClient; \
           print(MaestroClient().health())"
```

The client falls back to local Merkle generation if the gateway is unreachable —
both produce identical roots, so on-chain anchors verify either way.

## 2 — On-Chain Receipt Anchoring (Sepolia)

### 2a. Deploy the contract

**Before you begin — get testnet ETH (free):**
```
Alchemy Sepolia Faucet:  https://sepoliafaucet.com/             (0.5 ETH/day)
Infura Faucet:           https://www.infura.io/faucet/sepolia
Chainlink Faucet:        https://faucets.chain.link/sepolia
QuickNode Faucet:        https://faucet.quicknode.com/ethereum/sepolia
```
Note: some faucets require your mainnet wallet to hold ≥ 0.001 ETH (spam filter).
The deploy costs ~115 000 gas (≈ 0.002 SepoliaETH at 20 gwei).

**Get a free RPC URL:** Alchemy (`https://dashboard.alchemy.com/`) or Infura (`https://app.infura.io/`). Both offer a free tier for Sepolia.

**Get a free Etherscan API key** (for `--verify`): https://etherscan.io/register → API Keys.

```bash
# One-time setup (skip if you already have Foundry + onchain/lib/forge-std)
cd onchain
curl -L https://foundry.paradigm.xyz | bash && foundryup
# `forge-std` is already vendored at onchain/lib/forge-std; only run the next line on a fresh clone:
# forge install foundry-rs/forge-std --no-commit

# Run the test suite (unit + fuzz) first
forge test -vv

# Fill in D:\gemma4good\.env (see .env.example for all variables),
# then source it before running forge:
# Linux/macOS: source ../.env
# Windows PowerShell: Get-Content ..\.env | ForEach-Object { if ($_ -match '^([^#][^=]+)=(.*)$') { [System.Environment]::SetEnvironmentVariable($Matches[1], $Matches[2]) } }

# Or export manually:
export DEPLOYER_PRIVATE_KEY=0x...                                  # Sepolia-funded throwaway wallet
export SEPOLIA_RPC_URL=https://eth-sepolia.g.alchemy.com/v2/<key>
export ETHERSCAN_API_KEY=<key>                                     # optional, enables --verify

# Deploy (from onchain/ directory)
forge script script/Deploy.s.sol --rpc-url sepolia --broadcast --verify
```

The deployed address is printed at the end (`HAICAnchor deployed at: 0x...`). Save it to `HAIC_ANCHOR_ADDRESS` in your `.env`.

**Local alternative (no wallet, no faucet):**
```bash
# Starts Anvil, deploys, runs full anchor+verify roundtrip — takes ~5 seconds
cd D:\gemma4good && python onchain/live_roundtrip.py
# Result saved to: onchain/live_roundtrip_result.json
```

### 2b. Anchor a live receipt

```bash
pip install web3
export RPC_URL=$SEPOLIA_RPC_URL
export HAIC_ANCHOR_ADDRESS=0x<from-step-2a>
export DEPLOYER_PRIVATE_KEY=0x...

# Pull a receipt from the gateway (or use any existing local receipt JSON):
curl -X POST $MAESTRO_GATEWAY_BASE/v1/session/receipt \
  -H "Authorization: Bearer $MAESTRO_DEV_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"prod_001","messages":[{"role":"user","content":"hi"}],"consent":{"transcript":"granted"}}' \
  > receipt.json

# Dry run (no broadcast)
python -m onchain.anchor_client --receipt receipt.json --dry-run

# Anchor it
python -m onchain.anchor_client --receipt receipt.json --kind session
```

### 2c. Verify after the fact

```python
from onchain.anchor_client import AnchorClient
c = AnchorClient(rpc_url=..., contract_address=..., private_key=None)
print(c.verify("prod_001", "<merkle_root_hex>"))   # → True
print(c.get_anchor("prod_001"))                    # → full record
```

## 3 — Items Requiring Hardware We Don't Have

These were in Gemini's handoff but need GPU or external accounts to actually
execute. The repo already contains the runners; this is purely a "where to run"
note:

- **Live PRISM extraction + comparative model study (`v35-gov`)** —
  `experiments/temporal_signature.py` and `experiments/rank_geometry_sweep.py`
  are ready for Kaggle T4 / Colab A100 / Lambda. They write JSON output that
  `dashboard/app.py` already reads.
- **Incremental grounding LoRA training end-to-end** — same. The
  `incremental_grounding.run_grounding_update_handler` GPU paths are gated
  behind `try/except ImportError` and only run when torch+peft+transformers
  are present.
- **`.env` GOOGLE_API_KEY** — still blank. Rotate one in
  https://aistudio.google.com/app/apikey and write it into `.env` before
  running `_smoke_test_gemini.py`.

## 4 — Test Status

```
$ python -m pytest tests/
253 passed in ~3s
```

Counts by suite (verified via `pytest --collect-only`):

| Suite                              | Tests |
|------------------------------------|-------|
| test_anchor_client.py              |   12  |
| test_grounding_tracker.py          |   26  |
| test_haic_tools.py                 |   79  |
| test_incremental_grounding.py      |   46  |
| test_maestro_client.py             |    8  |
| test_maestro_gateway.py            |   12  |
| test_merkle.py                     |   23  |
| test_prism_client.py               |    9  |
| test_viability_condition.py        |   38  |
| **Total**                          |  253  |
