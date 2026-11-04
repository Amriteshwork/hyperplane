"""Toy model of YaRN's temperature: with s times more keys, the right key's share of attention
drops; multiplying every score by 1 + ln(s)/a wins it back.

Render: make videos paper=yarn scene=sharper-softmax
"""

import numpy as np
from manim import (
    BLUE,
    DOWN,
    UP,
    YELLOW,
    FadeIn,
    FadeOut,
    Rectangle,
    Scene,
    Text,
    Transform,
    VGroup,
)

LEAD = 5.0  # the right key's score; every other key scores 0
FEW, S = 8, 16
BASELINE = -1.9
HEIGHT = 3.4  # bar height for 100% of the attention


def chart(n: int, factor: float) -> tuple[VGroup, float]:
    """Two bars: the right key's share, and all n − 1 other keys' shares added together."""
    right = np.exp(LEAD * factor)
    share = right / (right + (n - 1))
    group = VGroup()
    for x, w, colour, name in [
        (-2.2, share, YELLOW, "the right key"),
        (2.2, 1 - share, BLUE, f"the other {n - 1} keys, together"),
    ]:
        h = max(0.02, HEIGHT * w)
        group.add(
            Rectangle(width=2.4, height=h, stroke_width=0)
            .set_fill(colour, 1)
            .move_to([x, BASELINE + h / 2, 0])
        )
        group.add(Text(name, font_size=22, color=colour).move_to([x, BASELINE - 0.35, 0]))
    return group, float(share)


def share_text(share: float) -> Text:
    return (
        Text(f"right key: {100 * share:.0f}% of the attention", font_size=28, color=YELLOW)
        .to_edge(UP)
        .shift(DOWN * 0.9)
    )


class SharperSoftmax(Scene):
    def construct(self) -> None:
        title = Text("More tokens, blurrier attention", font_size=34).to_edge(UP)
        bars, share = chart(FEW, 1.0)
        label = share_text(share)
        caption = Text(
            f"{FEW} keys: the right one (yellow) scores 5, the rest 0", font_size=24
        ).to_edge(DOWN)
        self.play(FadeIn(title), FadeIn(bars), FadeIn(label), FadeIn(caption))
        self.wait(1.5)

        many, share = chart(FEW * S, 1.0)
        self.play(
            Transform(bars, many),
            Transform(label, share_text(share)),
            Transform(
                caption,
                Text(
                    f"Context ×{S}: {FEW * S} keys share the same attention", font_size=24
                ).to_edge(DOWN),
            ),
            run_time=2,
        )
        self.wait(1.5)

        factor = 1 + np.log(S) / LEAD
        sharp, share = chart(FEW * S, factor)
        self.play(
            Transform(bars, sharp),
            Transform(label, share_text(share)),
            Transform(
                caption,
                Text(
                    f"Multiply every score by 1 + ln({S})/5 = {factor:.2f}: the lead is back",
                    font_size=24,
                    color=YELLOW,
                ).to_edge(DOWN),
            ),
            run_time=2,
        )
        self.wait(1.5)
        self.play(
            Transform(
                caption,
                Text(
                    "YaRN's fitted rule has the same shape: (0.1 ln s + 1)² = 1.63 at s = 16",
                    font_size=24,
                ).to_edge(DOWN),
            )
        )
        self.wait(2.5)
        self.play(FadeOut(*self.mobjects))
