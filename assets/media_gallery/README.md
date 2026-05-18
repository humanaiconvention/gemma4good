# media_gallery — owned by Ben's render pipeline

**Do not auto-render to this directory.**

The PNGs here are the canonical v3 renders produced by Ben on 2026-05-18 morning.
Parallel/intermediate renders (including a one-off Claude render from 2026-05-17
15:53) are preserved under `_archive/`:

- `_archive/v1_2026-05-16/` — earliest version
- `_archive/v2_2026-05-17/` — v2 before the Viability Condition restructure
- `_archive/v2_final_2026-05-17/` — late-v2 polish
- `_archive/claude_design_v3_2026-05-18/` — intermediate v3 from Ben's design pass
- `_archive/parallel_render_2026-05-18/` — the Claude one-off cairosvg render

The source SVGs Claude used for that one-off render now live under
`D:\gemma4good\video\_archive\graphics_v3_source_2026-05-18\` — moved out of
the active `video/graphics/` tree so they cannot be mistaken for a re-render
target. They are kept for reference only.

`video/assemble.py` writes its frames to `video/out/frames/` only — it does
not, and has never, written to this directory.

— 2026-05-18
