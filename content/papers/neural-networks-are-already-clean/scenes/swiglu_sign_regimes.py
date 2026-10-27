"""SwiGLU multiplies a gate by a value, so a neuron can fire positive for two different reasons.

Render: make videos paper=neural-networks-are-already-clean scene=swiglu-sign-regimes
"""

import numpy as np
from manim import (
    BLUE,
    DOWN,
    GREY_B,
    LEFT,
    RED,
    RIGHT,
    UP,
    YELLOW,
    Axes,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    Scene,
    SurroundingRectangle,
    Text,
    Transform,
    ValueTracker,
    VGroup,
    always_redraw,
)

#: (gate, value) pairs to visit: the ordinary + × + route, then the double negative.
ROUTES = [(2.0, 3.0), (-1.5, -4.0)]


def silu(g: float) -> float:
    return g / (1 + np.exp(-g))


class SwigluSignRegimes(Scene):
    def construct(self) -> None:
        title = Text("SwiGLU: output = SiLU(gate) × value", font_size=34).to_edge(UP)
        axes = Axes(x_range=[-5, 3, 1], y_range=[-1, 3, 1], x_length=6, y_length=3.6,
                    tips=False).to_edge(LEFT, buff=0.6).shift(DOWN * 0.3)  # fmt: skip
        curve = axes.plot(silu, x_range=[-5, 3], color=YELLOW)
        labels = VGroup(
            Text("gate", font_size=20, color=GREY_B).next_to(axes.x_axis, RIGHT, buff=0.1),
            Text("SiLU(gate)", font_size=20, color=YELLOW).next_to(axes.y_axis, UP, buff=0.1),
        )
        self.play(FadeIn(title), Create(axes), FadeIn(labels))
        self.play(Create(curve), run_time=1.5)

        # The dip: SiLU is mostly zero for negative gates, but not quite.
        dip = Dot(axes.c2p(-1.28, silu(-1.28)), color=BLUE)
        dip_note = Text("dips to about −0.28", font_size=20, color=BLUE).next_to(dip, DOWN)
        self.play(FadeIn(dip), FadeIn(dip_note))
        self.wait(1.5)

        gate, value = ValueTracker(ROUTES[0][0]), ValueTracker(ROUTES[0][1])
        marker = always_redraw(
            lambda: Dot(axes.c2p(gate.get_value(), silu(gate.get_value())), color=RED)
        )

        def readout() -> VGroup:
            g, v = gate.get_value(), value.get_value()
            return (
                VGroup(
                    Text(f"gate  = {g:+.1f}", font_size=26),
                    Text(f"SiLU  = {silu(g):+.2f}", font_size=26, color=YELLOW),
                    Text(f"value = {v:+.1f}", font_size=26),
                    Text(f"output = {silu(g) * v:+.2f}", font_size=30, color=RED),
                )
                .arrange(DOWN, aligned_edge=LEFT, buff=0.25)
                .to_edge(RIGHT, buff=0.8)
                .shift(UP * 0.4)
            )

        numbers = always_redraw(readout)
        caption = Text("Route 1: gate open (+), value positive (+).", font_size=24).to_edge(DOWN)
        self.play(FadeOut(dip_note), FadeIn(marker), FadeIn(numbers), FadeIn(caption))
        self.wait(2)

        self.play(
            Transform(caption, Text("Route 2: both negative. − × − is also positive.", font_size=24).to_edge(DOWN)),
            gate.animate.set_value(ROUTES[1][0]), value.animate.set_value(ROUTES[1][1]), run_time=2.5,
        )  # fmt: skip
        self.wait(1)
        box = SurroundingRectangle(numbers[-1], color=RED)
        self.play(Create(box))
        self.wait(1.5)

        table = (
            VGroup(
                Text("gate × value", font_size=22, color=GREY_B),
                Text("+ × +  →  strong +", font_size=22),
                Text("− × −  →  weak +   (the “echo”)", font_size=22, color=BLUE),
                Text("+ × −  →  −", font_size=22),
                Text("− × +  →  weak −", font_size=22),
            )
            .arrange(DOWN, aligned_edge=LEFT, buff=0.18)
            .to_edge(RIGHT, buff=0.6)
            .shift(DOWN * 0.2)
        )
        self.play(FadeOut(box), FadeOut(numbers), FadeIn(table))
        self.play(
            Transform(caption, Text("Two reasons to fire: split them, and the noise in the middle clears.",
                                    font_size=24, color=YELLOW).to_edge(DOWN))
        )  # fmt: skip
        self.wait(3)
        self.play(FadeOut(*self.mobjects))
