from pathlib import Path

import pytest

from scripts.render_videos import render_commands

CONTENT = Path(__file__).parents[3] / "content"
PAPER = "attention-is-all-you-need"


def test_renders_each_video_from_its_scene_file_and_class() -> None:
    [command] = render_commands(CONTENT, PAPER, "qkv-lookup", "l")
    assert command[-2:] == [
        str(CONTENT / "papers" / PAPER / "scenes" / "qkv_lookup.py"),
        "QkvLookup",
    ]
    assert "-ql" in command
    assert len(render_commands(CONTENT, PAPER, None, "h")) >= 2


def test_unknown_video_names_fail() -> None:
    with pytest.raises(SystemExit, match="no videos"):
        render_commands(CONTENT, PAPER, "nope", "h")
