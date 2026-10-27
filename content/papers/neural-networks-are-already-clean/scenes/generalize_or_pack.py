"""Squeeze a layer: generalization predicts coarser neurons, superposition predicts packed mixtures.

Render: make videos paper=neural-networks-are-already-clean scene=generalize-or-pack
"""

from manim import (
    BLUE,
    DOWN,
    GREEN,
    GREY_B,
    ORANGE,
    RED,
    RIGHT,
    UP,
    YELLOW,
    Dot,
    FadeIn,
    FadeOut,
    LaggedStart,
    ManimColor,
    RoundedRectangle,
    Scene,
    Text,
    Transform,
    VGroup,
    interpolate_color,
)

#: Four families of similar things (shades of one colour), three members each.
FAMILIES = [BLUE, GREEN, ORANGE, RED]
#: Which (family, member) each neuron holds, before and after squeezing.
GENERALIZE_4 = [[(f, m) for m in range(3)] for f in range(4)]
GENERALIZE_2 = [
    [(f, m) for f in (0, 1) for m in range(3)],
    [(f, m) for f in (2, 3) for m in range(3)],
]
PACK_4 = [
    [(0, 0), (2, 1), (3, 2)],
    [(1, 0), (3, 1), (0, 2)],
    [(2, 0), (0, 1), (1, 2)],
    [(3, 0), (1, 1), (2, 2)],
]
PACK_2 = [
    [(0, 0), (2, 1), (3, 2), (1, 0), (3, 1), (0, 2)],
    [(2, 0), (0, 1), (1, 2), (3, 0), (1, 1), (2, 2)],
]


def shade(family: int, member: int) -> ManimColor:
    return interpolate_color(ManimColor(FAMILIES[family]), ManimColor("#FFFFFF"), 0.18 * member)


def layer(neurons: list[list[tuple[int, int]]], width: float) -> VGroup:
    """One box per neuron, holding a dot for each thing it responds to."""
    boxes = VGroup()
    for things in neurons:
        dots = VGroup(*(Dot(radius=0.12, color=shade(f, m)) for f, m in things))
        dots.arrange_in_grid(cols=3, buff=0.15)
        box = RoundedRectangle(corner_radius=0.12, width=width, height=dots.height + 0.4)
        boxes.add(VGroup(box.set_stroke(GREY_B, 2).move_to(dots), dots))
    return boxes.arrange(DOWN, buff=0.25)


class GeneralizeOrPack(Scene):
    def construct(self) -> None:
        title = Text("Too few neurons: what gives?", font_size=36).to_edge(UP)
        heads = (
            VGroup(
                Text("generalization", font_size=26, color=YELLOW),
                Text("superposition", font_size=26, color=RED),
            )
            .arrange(RIGHT, buff=3.5)
            .next_to(title, DOWN, buff=0.3)
        )
        left = layer(GENERALIZE_4, 1.6).next_to(heads[0], DOWN, buff=0.3)
        right = layer(PACK_4, 1.6).next_to(heads[1], DOWN, buff=0.3)
        self.play(FadeIn(title), FadeIn(heads))
        self.play(LaggedStart(*(FadeIn(n) for n in [*left, *right]), lag_ratio=0.1))

        caption = Text("Same 12 things. Colours are families of similar things.", font_size=24)
        self.play(FadeIn(caption.to_edge(DOWN)))
        self.wait(1.5)
        self.play(
            Transform(caption, Text("Left groups similar things. Right packs things that rarely meet.",
                                    font_size=24).to_edge(DOWN))
        )  # fmt: skip
        self.wait(2)

        # Halve the neurons and retrain: the two ideas predict opposite results.
        squeeze = Text("Now give the layer half the neurons.", font_size=24, color=YELLOW).to_edge(
            DOWN
        )
        self.play(Transform(caption, squeeze))
        self.play(
            Transform(left, layer(GENERALIZE_2, 2.4).next_to(heads[0], DOWN, buff=0.4)),
            Transform(right, layer(PACK_2, 2.4).next_to(heads[1], DOWN, buff=0.4)),
            run_time=2,
        )
        verdicts = VGroup(
            Text("coarser, still readable", font_size=22, color=YELLOW).next_to(left, DOWN),
            Text("bigger mixtures", font_size=22, color=RED).next_to(right, DOWN),
        )
        self.play(FadeIn(verdicts))
        self.wait(2)

        # What the essay saw when it ran this test for real.
        result = Text(
            "InceptionV1 layer 3b, cut from 480 to 10 neurons:\n"
            "one “curves & loops” neuron instead of ten curve detectors.",
            font_size=22, line_spacing=1.2,
        ).to_edge(DOWN)  # fmt: skip
        self.play(Transform(caption, result), verdicts[1].animate.fade(0.7),
                  right.animate.fade(0.7), heads[1].animate.fade(0.7))  # fmt: skip
        self.play(left.animate.scale(1.05), heads[0].animate.scale(1.1))
        self.wait(3)
        self.play(FadeOut(*self.mobjects))
