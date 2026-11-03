"""Long-term decay: a RoPE score is a sum of arrows spinning at different speeds; far apart, they cancel.

Render: make videos paper=rope scene=spinning-arrows
"""

from itertools import pairwise

import numpy as np
from manim import (
    BLUE,
    DOWN,
    GREY_B,
    LEFT,
    RIGHT,
    UP,
    WHITE,
    YELLOW,
    Arrow,
    DashedLine,
    FadeIn,
    FadeOut,
    Scene,
    Text,
    Transform,
    ValueTracker,
    VGroup,
    always_redraw,
    linear,
)

PAIRS = 8
#: θᵢ = base^(−i/PAIRS): pair 0 turns 1 radian per token, each next pair slower.
SPEEDS = [100 ** (-i / PAIRS) for i in range(PAIRS)]
STEP = 0.6  # length of each pair's arrow
START = LEFT * 2.4 + DOWN * 0.6


def chain(distance: float) -> list[np.ndarray]:
    """Tips of the arrows laid tip to tail: Σ e^{i·distance·θᵢ}, one pair at a time."""
    points = [START]
    for speed in SPEEDS:
        a = distance * speed
        points.append(points[-1] + STEP * np.array([np.cos(a), np.sin(a), 0]))
    return points


class SpinningArrows(Scene):
    def construct(self) -> None:
        title = Text("Why far-away words whisper", font_size=34).to_edge(UP)
        d = ValueTracker(0)

        def arrows() -> VGroup:
            pts = chain(d.get_value())
            return VGroup(
                *(
                    Arrow(a, b, buff=0, color=YELLOW if i == 0 else BLUE, stroke_width=5)
                    for i, (a, b) in enumerate(pairwise(pts))
                )
            )

        def total() -> DashedLine:
            pts = chain(d.get_value())
            return DashedLine(pts[0], pts[-1], color=WHITE, stroke_width=3)

        def readout() -> Text:
            pts = chain(d.get_value())
            size = np.linalg.norm(pts[-1] - pts[0]) / (STEP * PAIRS)
            return Text(
                f"distance {d.get_value():5.1f}     size of the sum: {100 * size:3.0f}%",
                font_size=26,
            ).next_to(title, DOWN, buff=0.35)

        legend = (
            VGroup(
                Text("one arrow per pair of dimensions", font_size=20, color=BLUE),
                Text("fastest pair", font_size=20, color=YELLOW),
                Text("dashed: their sum (the score's size)", font_size=20, color=GREY_B),
            )
            .arrange(DOWN, aligned_edge=LEFT)
            .to_corner(DOWN + RIGHT)
        )
        self.play(FadeIn(title), FadeIn(legend))
        self.add(always_redraw(arrows), always_redraw(total), always_redraw(readout))
        caption = Text("Distance 0: every clock agrees, the arrows line up", font_size=24)
        caption.next_to(title, DOWN, buff=0.9)
        self.play(FadeIn(caption))
        self.wait(1.5)

        self.play(
            Transform(
                caption,
                Text("Move apart: each arrow turns at its own speed", font_size=24).move_to(
                    caption
                ),
            ),
        )
        self.play(d.animate.set_value(6), run_time=5, rate_func=linear)
        self.play(
            Transform(
                caption,
                Text("They point every which way, and cancel", font_size=24).move_to(caption),
            ),
        )
        self.play(d.animate.set_value(60), run_time=10, rate_func=linear)
        self.wait(1)
        self.play(
            Transform(
                caption,
                Text(
                    "Close words can score loudly; distant ones mostly can't",
                    font_size=24,
                    color=YELLOW,
                ).move_to(caption),
            ),
        )
        self.play(d.animate.set_value(0), run_time=3)
        self.wait(2)
        self.play(FadeOut(*self.mobjects))
