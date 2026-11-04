"""A previous-token head from RoPE's fast clocks: every key is the same arrow turned by its
position; the query is turned back one notch, so it lines up with the key one step back. One
clock alone also half-matches word 0 (6 rad is nearly a full turn); a second clock fixes it.

Render: make videos paper=rope scene=previous-token-head
"""

import numpy as np
from manim import (
    BLUE,
    DOWN,
    GREY_B,
    UP,
    YELLOW,
    Arrow,
    Circle,
    FadeIn,
    FadeOut,
    GrowFromEdge,
    LaggedStart,
    Rectangle,
    Scene,
    Text,
    Transform,
    VGroup,
)

KEYS = 8  # positions 0…7; the query sits at position 7
QUERY_POS = KEYS - 1
SPEED = 1.0  # radians per position: RoPE's fastest clock
SECOND = 10000 ** (-2 / 128)  # the next clock along, ≈ 0.87 rad per position
SHARPNESS = 6.0  # a long query and key make softmax decisive
RADIUS = 0.55


def unit(angle: float) -> np.ndarray:
    return np.array([np.cos(angle), np.sin(angle), 0])


class PreviousTokenHead(Scene):
    def construct(self) -> None:
        title = Text("A previous-token head, built from the fast clocks", font_size=32).to_edge(UP)
        dials = VGroup(*(Circle(RADIUS).set_stroke(GREY_B, 1.5) for _ in range(KEYS)))
        dials.arrange(buff=0.35).shift(UP * 0.9)
        labels = VGroup(
            *(
                Text(f"{j}", font_size=22, color=GREY_B).next_to(d, UP, buff=0.15)
                for j, d in enumerate(dials)
            )
        )
        keys = VGroup(
            *(
                Arrow(
                    d.get_center(),
                    d.get_center() + RADIUS * unit(j * SPEED),
                    buff=0,
                    color=BLUE,
                    stroke_width=5,
                )
                for j, d in enumerate(dials)
            )
        )
        self.play(FadeIn(title), FadeIn(dials), FadeIn(labels))
        caption = Text("Every key is the same arrow, turned 1 radian per position", font_size=24)
        caption.to_edge(DOWN)
        self.play(FadeIn(caption))
        self.play(LaggedStart(*(FadeIn(k) for k in keys), lag_ratio=0.3), run_time=3)
        self.wait(0.5)

        # The query at position 7 is the same arrow turned back one notch first: angle (7 − 1) rad.
        query_angle = (QUERY_POS - 1) * SPEED
        self.play(
            Transform(
                caption,
                Text(
                    "The query (word 7) is turned back one notch: it points like word 6",
                    font_size=24,
                    color=YELLOW,
                ).to_edge(DOWN),
            )
        )
        ghosts = VGroup(
            *(
                Arrow(
                    d.get_center(),
                    d.get_center() + RADIUS * unit(query_angle),
                    buff=0,
                    color=YELLOW,
                    stroke_width=4,
                )
                for d in dials
            )
        )
        self.play(LaggedStart(*(FadeIn(g) for g in ghosts), lag_ratio=0.15), run_time=2.5)
        self.wait(0.5)

        def softmax(scores: np.ndarray) -> np.ndarray:
            w = np.exp(SHARPNESS * scores)
            return w / w.sum()

        gaps = QUERY_POS - 1 - np.arange(KEYS)
        weights = softmax(np.cos(gaps * SPEED))
        both = softmax(np.cos(gaps * SPEED) + np.cos(gaps * SECOND))
        baseline = -2.3  # bars grow up from here, below the dials

        def bar_chart(w: np.ndarray) -> VGroup:
            return VGroup(
                *(
                    Rectangle(width=0.5, height=max(0.03, 2.0 * x), stroke_width=0)
                    .set_fill(YELLOW if j == QUERY_POS - 1 else BLUE, 1)
                    .move_to([d.get_x(), baseline + max(0.03, 2.0 * x) / 2, 0])
                    for j, (d, x) in enumerate(zip(dials, w, strict=True))
                )
            )

        bars = bar_chart(weights)
        self.play(
            Transform(
                caption,
                Text("Attention = softmax of how well each pair lines up", font_size=24).to_edge(
                    DOWN
                ),
            ),
            LaggedStart(*(GrowFromEdge(b, DOWN) for b in bars), lag_ratio=0.1),
            run_time=2.5,
        )
        share = Text(
            f"{100 * weights[QUERY_POS - 1]:.0f}% of the attention on word 6",
            font_size=26,
            color=YELLOW,
        )
        share.move_to([0, baseline - 0.45, 0])
        self.play(FadeIn(share))
        self.wait(1.5)
        self.play(
            Transform(
                caption,
                Text(
                    "Word 0 lights up too: 6 radians is almost a full turn",
                    font_size=24,
                    color=BLUE,
                ).to_edge(DOWN),
            )
        )
        self.wait(2)
        self.play(
            Transform(
                caption,
                Text(
                    "Add the next clock (0.87 rad per position): they never both come round",
                    font_size=24,
                ).to_edge(DOWN),
            ),
            Transform(bars, bar_chart(both)),
            Transform(
                share,
                Text(
                    f"{100 * both[QUERY_POS - 1]:.0f}% of the attention on word 6",
                    font_size=26,
                    color=YELLOW,
                ).move_to(share),
            ),
            run_time=2,
        )
        self.wait(2)
        self.play(
            Transform(
                caption,
                Text(
                    "Same rule at every position: a head that always reads the previous word",
                    font_size=24,
                    color=YELLOW,
                ).to_edge(DOWN),
            )
        )
        self.wait(2.5)
        self.play(FadeOut(*self.mobjects))
