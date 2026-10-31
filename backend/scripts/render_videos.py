"""Render a page's Manim scenes to MP4 for uploading to YouTube (ADR 0022).

Needs the `videos` group and cairo/pango headers (Debian/Ubuntu:
`sudo apt install libcairo2-dev libpango1.0-dev pkg-config`). Output goes to the git-ignored
`build/videos/`; once uploaded, put each video's id in the page's `videos.<name>.youtube`.

Usage: uv run --group videos python -m scripts.render_videos PAGE [--concept] [--scene NAME]
"""

import argparse
import subprocess
import sys
from pathlib import Path

from app.content.catalog import ContentCatalog, scene_class
from app.core.config import get_settings

OUT = Path(__file__).parents[2] / "build" / "videos"


def render_commands(
    content: Path, page_id: str, scene: str | None, quality: str, *, concept: bool = False
) -> list[list[str]]:
    catalog = ContentCatalog.load(content)
    videos = catalog.concept(page_id).meta.videos if concept else catalog.paper(page_id).meta.videos
    names = [scene] if scene else list(videos)
    if unknown := [n for n in names if n not in videos]:
        raise SystemExit(f"{page_id} has no videos {unknown}; it has {list(videos)}")
    scenes = content / ("concepts" if concept else "papers") / page_id / "scenes"
    return [
        [
            sys.executable, "-m", "manim", "render", f"-q{quality}",
            "--media_dir", str(OUT / page_id),
            str(scenes / f"{name.replace('-', '_')}.py"), scene_class(name),
        ]
        for name in names
    ]  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("page", help="a paper id, or a concept id with --concept")
    parser.add_argument("--concept", action="store_true", help="the page is a concept")
    parser.add_argument("--scene", help="one video name; default: all of the page's videos")
    parser.add_argument("--quality", default="h", choices=list("lmhpk"), help="manim -q flag")
    parser.add_argument("--content-dir", type=Path, default=get_settings().content_dir)
    args = parser.parse_args(argv)
    commands = render_commands(
        args.content_dir, args.page, args.scene, args.quality, concept=args.concept
    )
    for command in commands:
        subprocess.run(command, check=True)
    print(f"rendered into {OUT / args.page}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
