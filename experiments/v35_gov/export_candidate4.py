from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_ROOT = Path(
    os.environ.get(
        "GEMMA4GOOD_ARCHIVE_ROOT",
        REPO_ROOT / "artifacts" / "v35_gov_archive_2026-04-21",
    )
)
DEFAULT_CANONICAL_DIR = ARCHIVE_ROOT / "canonical_candidate_4"
DEFAULT_CANDIDATE_DIR = ARCHIVE_ROOT / "candidate_4_indexerror_loss_0_4645"
DEFAULT_SCAFFOLD_DIR = ARCHIVE_ROOT / "adapter_scaffold"
DEFAULT_RESULTS_JSON = ARCHIVE_ROOT / "final_results" / "haic_v35_gov_full_results.json"
DEFAULT_BASE_ROOT = Path(
    os.environ.get(
        "GEMMA4GOOD_BASE_ROOT",
        REPO_ROOT / "artifacts" / "models--google--gemma-4-E2B-it",
    )
)
DEFAULT_LLAMA_CPP = Path(os.environ.get("LLAMA_CPP_ROOT", REPO_ROOT / "artifacts" / "llama.cpp"))
DEFAULT_GGUF_DIR = Path(os.environ.get("GEMMA4GOOD_GGUF_DIR", REPO_ROOT / "artifacts" / "gguf"))


def read_main_snapshot(base_root: Path) -> Path:
    refs_main = base_root / "refs" / "main"
    if not refs_main.exists():
        raise FileNotFoundError(f"Missing refs/main under {base_root}")
    snapshot_id = refs_main.read_text(encoding="utf-8").strip()
    snapshot_dir = base_root / "snapshots" / snapshot_id
    if not snapshot_dir.exists():
        raise FileNotFoundError(f"Snapshot {snapshot_id} not found under {base_root / 'snapshots'}")
    return snapshot_dir


def ensure_canonical_dir(canonical_dir: Path, candidate_dir: Path, scaffold_dir: Path, results_json: Path) -> Path:
    adapter_dir = canonical_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    copy_pairs = [
        (scaffold_dir / "adapter_config.json", adapter_dir / "adapter_config.json"),
        (scaffold_dir / "chat_template.jinja", adapter_dir / "chat_template.jinja"),
        (scaffold_dir / "README.md", adapter_dir / "README.md"),
        (scaffold_dir / "tokenizer.json", adapter_dir / "tokenizer.json"),
        (scaffold_dir / "tokenizer_config.json", adapter_dir / "tokenizer_config.json"),
        (candidate_dir / "adapter_model.safetensors", adapter_dir / "adapter_model.safetensors"),
        (candidate_dir / "trainer_state.json", canonical_dir / "trainer_state.json"),
        (candidate_dir / "unsloth.log", canonical_dir / "unsloth.log"),
        (results_json, canonical_dir / "haic_v35_gov_full_results.json"),
    ]

    for src, dst in copy_pairs:
        if not src.exists():
            raise FileNotFoundError(f"Missing required source file: {src}")
        shutil.copy2(src, dst)

    manifest = {
        "label": "v35-gov canonical candidate 4",
        "source_candidate_dir": str(candidate_dir),
        "adapter_dir": str(adapter_dir),
        "results_json": str(canonical_dir / "haic_v35_gov_full_results.json"),
    }
    (canonical_dir / "canonical_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return adapter_dir


def run_checked(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"\n>>> {' '.join(str(part) for part in cmd)}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge and quantize the canonical v35-gov candidate 4 adapter.")
    parser.add_argument("--canonical-dir", type=Path, default=DEFAULT_CANONICAL_DIR)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--scaffold-dir", type=Path, default=DEFAULT_SCAFFOLD_DIR)
    parser.add_argument("--results-json", type=Path, default=DEFAULT_RESULTS_JSON)
    parser.add_argument("--base-root", type=Path, default=DEFAULT_BASE_ROOT)
    parser.add_argument("--llama-cpp", type=Path, default=DEFAULT_LLAMA_CPP)
    parser.add_argument("--gguf-dir", type=Path, default=DEFAULT_GGUF_DIR)
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--skip-quantize", action="store_true")
    args = parser.parse_args()

    canonical_dir = args.canonical_dir
    merged_dir = canonical_dir / "merged"
    f16_path = canonical_dir / "haic-gemma4-v35-gov-F16.gguf"
    q5_path = args.gguf_dir / "haic-gemma4-v35-gov-Q5_K_M.gguf"

    adapter_dir = ensure_canonical_dir(canonical_dir, args.candidate_dir, args.scaffold_dir, args.results_json)
    snapshot_dir = read_main_snapshot(args.base_root)
    print(f"Using base snapshot: {snapshot_dir}")
    print(f"Using adapter dir  : {adapter_dir}")

    if not args.skip_merge:
        from transformers import AutoTokenizer
        try:
            from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration
        except Exception:
            from transformers import Gemma4ForConditionalGeneration
        from peft import PeftModel

        merged_dir.mkdir(parents=True, exist_ok=True)
        print("\nLoading base model on CPU for merge...")
        base_model = Gemma4ForConditionalGeneration.from_pretrained(
            str(snapshot_dir),
            torch_dtype="auto",
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(str(snapshot_dir))
        model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        model = model.merge_and_unload()
        model.eval()
        print("Saving merged safetensors...")
        model.save_pretrained(str(merged_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(merged_dir))

        export_manifest = {
            "base_snapshot": str(snapshot_dir),
            "adapter_dir": str(adapter_dir),
            "merged_dir": str(merged_dir),
        }
        (merged_dir / "export_manifest.json").write_text(json.dumps(export_manifest, indent=2), encoding="utf-8")

    if not args.skip_convert:
        convert_script = args.llama_cpp / "convert_hf_to_gguf.py"
        if not convert_script.exists():
            raise FileNotFoundError(f"Missing convert_hf_to_gguf.py at {convert_script}")
        run_checked(
            [
                sys.executable,
                str(convert_script),
                str(merged_dir),
                "--outfile",
                str(f16_path),
                "--outtype",
                "f16",
            ],
            cwd=args.llama_cpp,
        )

    if not args.skip_quantize:
        quantize_bin = args.llama_cpp / "build" / "bin" / "llama-quantize.exe"
        if not quantize_bin.exists():
            raise FileNotFoundError(f"Missing llama-quantize.exe at {quantize_bin}")
        args.gguf_dir.mkdir(parents=True, exist_ok=True)
        run_checked([str(quantize_bin), str(f16_path), str(q5_path), "Q5_K_M"], cwd=args.llama_cpp)

        promotion_manifest = {
            "canonical_dir": str(canonical_dir),
            "adapter_dir": str(adapter_dir),
            "merged_dir": str(merged_dir),
            "f16_path": str(f16_path),
            "q5_path": str(q5_path),
            "status": "quantized",
        }
        (canonical_dir / "promotion_candidate_manifest.json").write_text(
            json.dumps(promotion_manifest, indent=2),
            encoding="utf-8",
        )

    print("\nExport flow completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
