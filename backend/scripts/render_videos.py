"""Render a paper's Manim scenes to MP4 for uploading to YouTube (ADR 0022).

Needs the `videos` group and cairo/pango headers (Debian/Ubuntu:
`sudo apt install libcairo2-dev libpango1.0-dev pkg-config`). Output goes to the git-ignored
`build/videos/`; once uploaded, put each video's id in the paper's `videos.<name>.youtube`.

Usage: uv run --group videos python -m scripts.render_videos PAPER [--scene NAME] [--quality h]
"""

import argparse
import subprocess
import sys
from pathlib import Path

from app.content.catalog import ContentCatalog, scene_class
from app.core.config import get_settings

OUT = Path(__file__).parents[2] / "build" / "videos"


def render_commands(
    content: Path, paper_id: str, scene: str | None, quality: str
) -> list[list[str]]:
    paper = ContentCatalog.load(content).paper(paper_id)
    names = [scene] if scene else list(paper.meta.videos)
    if unknown := [n for n in names if n not in paper.meta.videos]:
        raise SystemExit(f"{paper_id} has no videos {unknown}; it has {list(paper.meta.videos)}")
    scenes = content / "papers" / paper_id / "scenes"
    return [
        [
            sys.executable, "-m", "manim", "render", f"-q{quality}",
            "--media_dir", str(OUT / paper_id),
            str(scenes / f"{name.replace('-', '_')}.py"), scene_class(name),
        ]
        for name in names
    ]  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("paper")
    parser.add_argument("--scene", help="one video name; default: all of the paper's videos")
    parser.add_argument("--quality", default="h", choices=list("lmhpk"), help="manim -q flag")
    parser.add_argument("--content-dir", type=Path, default=get_settings().content_dir)
    args = parser.parse_args(argv)
    for command in render_commands(args.content_dir, args.paper, args.scene, args.quality):
        subprocess.run(command, check=True)
    print(f"rendered into {OUT / args.paper}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
