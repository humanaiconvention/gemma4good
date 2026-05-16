#!/usr/bin/env python3
"""
assemble.py — build the 2:30 HAIC Gemma 4 Good submission video from
SVG shot frames + per-segment VO recordings.

This is the lo-fi pipeline. Hard cuts everywhere, no crossfades, no motion.
For a polished cut, use After Effects with animation_spec.json + the SVGs.

Usage:
    python assemble.py
    python assemble.py --vo-dir D:\\gemma4good\\video\\vo --out final.mp4
    python assemble.py --kaggle-capture D:\\gemma4good\\video\\kaggle_repro.mkv

Inputs (defaults are repo-relative):
    graphics\\shot_01_logo_lockup.svg
    graphics\\shot_03_cold_open.svg
    graphics\\shot_04_five_tools.svg
    graphics\\shot_05_scenarios.svg
    graphics\\shot_09_split_proof.svg
    graphics\\shot_16_frontier.svg
    graphics\\shot_19_spec_url.svg
    graphics\\shot_20_tag_plate.svg
    vo\\seg_01.wav   (or .m4a — picked take, renamed by you)
    vo\\seg_02.wav
    vo\\seg_03.wav
    vo\\seg_04.wav
    vo\\seg_05.wav
    vo\\seg_06.wav
    [optional] kaggle_repro.mkv — the live screen capture

Outputs:
    out\\frames\\*.png        — PNG renders of each SVG at 1920x1080
    out\\audio_track.wav      — concatenated VO at 2:30 total
    out\\haic_gemma4_good.mp4 — final muxed video

Requires:
    ffmpeg on PATH
    One SVG renderer on PATH: inkscape (preferred) | rsvg-convert | or
        Python `cairosvg` (pip install cairosvg)
"""

from __future__ import annotations
import argparse, shutil, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_GRAPHICS = HERE / "graphics"
DEFAULT_VO = HERE / "vo"
DEFAULT_OUT = HERE / "out"

# Shot timeline (seconds). Total = 150 s = 2:30.
SHOTS = [
    # (svg basename or special, in_s, out_s, label)
    ("shot_01_logo_lockup.svg",  0.0,   5.0,  "logo lockup"),
    ("shot_03_cold_open.svg",    5.0,  14.0,  "cold open line"),
    ("shot_04_five_tools.svg",  14.0,  22.0,  "five tools"),
    ("shot_05_scenarios.svg",   22.0,  46.0,  "three scenarios"),
    ("shot_09_split_proof.svg", 46.0,  84.0,  "split proof"),
    ("__kaggle__",              84.0, 110.0,  "kaggle capture"),
    ("shot_16_frontier.svg",   110.0, 138.0,  "frontier pitch"),
    ("shot_19_spec_url.svg",   138.0, 145.0,  "spec url"),
    ("shot_20_tag_plate.svg",  145.0, 150.0,  "tag plate"),
]

# VO segment durations (seconds). Sum should equal 150 s after silence pads.
VO_SEGMENTS = [
    ("seg_01", 14.0, 0.0),   # cold open: starts at 0:00 with 0 lead silence
    ("seg_02", 32.0, 0.0),   # what it is: starts at 0:14
    ("seg_03", 38.0, 0.0),   # proof beat: starts at 0:46
    ("seg_04", 26.0, 0.0),   # reproducibility: starts at 1:24
    ("seg_05", 35.0, 0.0),   # frontier pitch: starts at 1:50
    ("seg_06",  5.0, 0.0),   # tag: starts at 2:25
]

WIDTH, HEIGHT, FPS = 1920, 1080, 24


# ----------------------------------------------------------------------------
# SVG -> PNG rendering
# ----------------------------------------------------------------------------

def detect_renderer() -> tuple[str, list[str]]:
    """Return (name, base_cmd) of an available SVG renderer."""
    if shutil.which("inkscape"):
        return "inkscape", ["inkscape"]
    if shutil.which("rsvg-convert"):
        return "rsvg-convert", ["rsvg-convert"]
    try:
        import cairosvg  # noqa: F401
        return "cairosvg", []
    except ImportError:
        pass
    sys.exit("No SVG renderer found. Install one of:\n"
             "  - Inkscape (https://inkscape.org), or\n"
             "  - librsvg's rsvg-convert, or\n"
             "  - Python cairosvg: pip install cairosvg")


def render_svg(svg: Path, png: Path, renderer: tuple[str, list[str]]) -> None:
    name, base = renderer
    png.parent.mkdir(parents=True, exist_ok=True)
    if name == "inkscape":
        subprocess.run(base + [
            "--export-type=png",
            f"--export-filename={png}",
            f"--export-width={WIDTH}",
            f"--export-height={HEIGHT}",
            str(svg),
        ], check=True)
    elif name == "rsvg-convert":
        subprocess.run(base + [
            "-w", str(WIDTH), "-h", str(HEIGHT),
            "-o", str(png), str(svg),
        ], check=True)
    elif name == "cairosvg":
        import cairosvg
        cairosvg.svg2png(url=str(svg), write_to=str(png),
                         output_width=WIDTH, output_height=HEIGHT)


# ----------------------------------------------------------------------------
# Video segment builders
# ----------------------------------------------------------------------------

def build_still_clip(png: Path, duration_s: float, out_mp4: Path) -> None:
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-loop", "1", "-i", str(png),
        "-t", f"{duration_s}",
        "-r", str(FPS),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-vf", f"scale={WIDTH}:{HEIGHT}",
        str(out_mp4),
    ], check=True)


def build_video_passthrough(src: Path, duration_s: float, out_mp4: Path) -> None:
    """Take a screen-capture video (the Kaggle take) and trim/loop to fit."""
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src),
        "-t", f"{duration_s}",
        "-r", str(FPS),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-vf", f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,"
                f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=0x0B0B0F",
        "-an",
        str(out_mp4),
    ], check=True)


def build_placeholder_clip(label: str, duration_s: float, out_mp4: Path) -> None:
    """If no Kaggle capture provided, drop a label placeholder card."""
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", f"color=0x0B0B0F:s={WIDTH}x{HEIGHT}:r={FPS}",
        "-t", f"{duration_s}",
        "-vf", (f"drawtext=text='[ {label} placeholder ]':fontcolor=white:fontsize=42:"
                f"x=(w-text_w)/2:y=(h-text_h)/2"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(out_mp4),
    ], check=True)


def concat_clips(clip_paths: list[Path], out_mp4: Path) -> None:
    list_file = out_mp4.parent / "concat_list.txt"
    list_file.write_text("\n".join(f"file '{p.as_posix()}'" for p in clip_paths))
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(out_mp4),
    ], check=True)


# ----------------------------------------------------------------------------
# Audio assembly
# ----------------------------------------------------------------------------

def find_vo_file(vo_dir: Path, base: str) -> Path | None:
    for ext in (".wav", ".m4a", ".mp3", ".flac"):
        p = vo_dir / f"{base}{ext}"
        if p.exists():
            return p
    return None


def build_audio_track(vo_dir: Path, out_wav: Path) -> None:
    """Concatenate the 6 VO segments into a single 150 s audio track,
    padding each segment with silence to fit its target duration.

    If a segment is missing, that slot is silence."""
    inputs, filt = [], []
    for idx, (base, target_s, _) in enumerate(VO_SEGMENTS):
        src = find_vo_file(vo_dir, base)
        if src is None:
            # Silent placeholder for this slot
            filt.append(f"anullsrc=channel_layout=mono:sample_rate=48000,atrim=0:{target_s}[a{idx}];")
        else:
            inputs.extend(["-i", str(src)])
            filt.append(
                f"[{len(inputs)//2 - 1}:a]aresample=48000,aformat=channel_layouts=mono,"
                f"apad=whole_dur={target_s},atrim=0:{target_s}[a{idx}];"
            )
    # Join all aN streams in sequence
    join = "".join(f"[a{i}]" for i in range(len(VO_SEGMENTS)))
    filter_complex = "".join(filt) + f"{join}concat=n={len(VO_SEGMENTS)}:v=0:a=1[out]"
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    # If no real inputs, supply a dummy
    if not inputs:
        cmd.extend(["-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=48000"])
    cmd.extend(inputs)
    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:a", "pcm_s24le",
        str(out_wav),
    ])
    subprocess.run(cmd, check=True)


def mux_av(video: Path, audio: Path, out_mp4: Path) -> None:
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video), "-i", str(audio),
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(out_mp4),
    ], check=True)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--graphics-dir", type=Path, default=DEFAULT_GRAPHICS)
    p.add_argument("--vo-dir", type=Path, default=DEFAULT_VO)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT / "haic_gemma4_good.mp4")
    p.add_argument("--kaggle-capture", type=Path, default=None,
                   help="Optional path to the Kaggle screen recording (mkv/mp4).")
    args = p.parse_args()

    out_dir = args.out.parent
    frames_dir = out_dir / "frames"
    clips_dir = out_dir / "clips"
    frames_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    renderer = detect_renderer()
    print(f"[1/4] Rendering SVGs with {renderer[0]} ...")
    for spec in SHOTS:
        svg_name, _, _, _ = spec
        if svg_name == "__kaggle__":
            continue
        svg = args.graphics_dir / svg_name
        if not svg.exists():
            sys.exit(f"missing SVG: {svg}")
        png = frames_dir / svg.with_suffix(".png").name
        if not png.exists() or png.stat().st_mtime < svg.stat().st_mtime:
            render_svg(svg, png, renderer)

    print(f"[2/4] Building shot clips ...")
    clip_paths: list[Path] = []
    for svg_name, t_in, t_out, label in SHOTS:
        dur = t_out - t_in
        clip = clips_dir / f"{label.replace(' ', '_')}.mp4"
        if svg_name == "__kaggle__":
            if args.kaggle_capture and args.kaggle_capture.exists():
                build_video_passthrough(args.kaggle_capture, dur, clip)
            else:
                build_placeholder_clip("kaggle screen capture", dur, clip)
        else:
            png = frames_dir / svg_name.replace(".svg", ".png")
            build_still_clip(png, dur, clip)
        clip_paths.append(clip)

    video_silent = out_dir / "video_silent.mp4"
    concat_clips(clip_paths, video_silent)

    print(f"[3/4] Building audio track from {args.vo_dir} ...")
    audio_track = out_dir / "audio_track.wav"
    build_audio_track(args.vo_dir, audio_track)

    print(f"[4/4] Muxing -> {args.out}")
    mux_av(video_silent, audio_track, args.out)
    print("done.")


if __name__ == "__main__":
    main()
