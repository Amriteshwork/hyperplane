"""The 'cat or car' neuron is one V detector: pointed things push it up, round things push it down.

Render: make videos paper=neural-networks-are-already-clean scene=v-not-cat-car
"""

from manim import (
    BLUE,
    DOWN,
    GREY_B,
    ORANGE,
    RED,
    UP,
    WHITE,
    YELLOW,
    Circle,
    Create,
    FadeIn,
    FadeOut,
    GrowFromEdge,
    LaggedStart,
    Line,
    Polygon,
    Rectangle,
    Scene,
    Text,
    Transform,
    VGroup,
)

#: Toy activations for the illustration; the essay's figure shows the real ones.
ACTS = {"cat face": 0.9, "car front": 0.85, "flame": 0.8, "fox face": 0.75, "an O": -0.6}
BAR_UNIT = 1.6
BASELINE_Y = -2.0


def cat() -> VGroup:
    head = Polygon([-0.7, 0.5, 0], [-0.5, 1.0, 0], [-0.25, 0.55, 0], [0.25, 0.55, 0],
                   [0.5, 1.0, 0], [0.7, 0.5, 0], [0, -0.6, 0])  # fmt: skip
    eyes = VGroup(Circle(0.07).shift([-0.25, 0.25, 0]), Circle(0.07).shift([0.25, 0.25, 0]))
    return VGroup(head.set_stroke(WHITE, 3), eyes.set_fill(WHITE, 1).set_stroke(width=0))


def car() -> VGroup:
    body = Polygon([-0.8, 0.6, 0], [0.8, 0.6, 0], [0.5, -0.2, 0], [0, -0.6, 0], [-0.5, -0.2, 0])
    lights = VGroup(Circle(0.1).shift([-0.5, 0.35, 0]), Circle(0.1).shift([0.5, 0.35, 0]))
    return VGroup(body.set_stroke(WHITE, 3), lights.set_stroke(YELLOW, 2))


def flame() -> VGroup:
    tongue = Polygon([-0.5, 0.8, 0], [-0.15, 0.3, 0], [0, 1.0, 0], [0.15, 0.3, 0], [0.5, 0.8, 0],
                     [0, -0.6, 0])  # fmt: skip
    return VGroup(tongue.set_stroke(ORANGE, 3))


def fox() -> VGroup:
    head = Polygon([-0.8, 0.9, 0], [-0.3, 0.45, 0], [0.3, 0.45, 0], [0.8, 0.9, 0], [0, -0.6, 0])
    return VGroup(head.set_stroke(RED, 3))


def ring() -> VGroup:
    return VGroup(Circle(0.6).set_stroke(BLUE, 4))


def v_mark(shape: VGroup) -> VGroup:
    """The V the neuron actually looks for, drawn over the shape's pointed lower half."""
    top, bottom = shape.get_top()[1] - 0.2, shape.get_bottom()[1]
    x = shape.get_center()[0]
    return VGroup(
        Line([x - 0.6, top, 0], [x, bottom, 0]), Line([x + 0.6, top, 0], [x, bottom, 0])
    ).set_stroke(YELLOW, 6)


class VNotCatCar(Scene):
    def construct(self) -> None:
        title = Text("InceptionV1 neuron 4e:55: cats OR cars?", font_size=34).to_edge(UP)
        shapes = VGroup(cat(), car(), flame(), fox(), ring()).arrange(buff=1.0).shift(UP * 0.9)
        labels = VGroup(
            *(Text(name, font_size=20, color=GREY_B).next_to(s, DOWN, buff=0.2)
              for s, name in zip(shapes, ACTS))
        )  # fmt: skip
        axis = Line([-6.5, BASELINE_Y, 0], [6.5, BASELINE_Y, 0]).set_stroke(GREY_B, 2)
        bars = VGroup()
        for shape, act in zip(shapes, ACTS.values()):
            bar = Rectangle(width=0.6, height=abs(act) * BAR_UNIT).set_stroke(width=0)
            bar.set_fill(YELLOW if act > 0 else BLUE, 0.85)
            edge = DOWN if act > 0 else UP
            bar.move_to([shape.get_center()[0], BASELINE_Y, 0], aligned_edge=edge)
            bars.add(bar)

        # The famous reading: two unrelated things.
        self.play(FadeIn(title), FadeIn(shapes[:2]), FadeIn(labels[:2]), Create(axis))
        self.play(*(GrowFromEdge(b, DOWN) for b in bars[:2]))
        caption = Text("It fires for both, so it must be a mixture… right?", font_size=24)
        self.play(FadeIn(caption.to_edge(DOWN)))
        self.wait(1.5)

        # Ask what one thing they share.
        marks = VGroup(*(v_mark(s) for s in shapes[:4]))
        self.play(
            Transform(caption, Text("What one shape do they share?", font_size=24).to_edge(DOWN)),
            LaggedStart(*(Create(m) for m in marks[:2]), lag_ratio=0.4),
        )
        self.wait(1)

        # Its other strong inputs are V-shaped too, and its negative pole is round.
        self.play(
            Transform(caption, Text("Flames and fox faces drive it too: all pointed.", font_size=24).to_edge(DOWN)),
            FadeIn(shapes[2:4]), FadeIn(labels[2:4]),
            *(GrowFromEdge(b, DOWN) for b in bars[2:4]),
            LaggedStart(*(Create(m) for m in marks[2:]), lag_ratio=0.4),
        )  # fmt: skip
        self.wait(1.5)
        self.play(
            Transform(caption, Text("And an O, round where the V is pointed, pushes it negative.", font_size=24).to_edge(DOWN)),
            FadeIn(shapes[4]), FadeIn(labels[4]), GrowFromEdge(bars[4], UP),
        )  # fmt: skip
        self.wait(2)

        verdict = Text("Not OR(cat, car): one V detector, with an O on its negative side.",
                       font_size=26, color=YELLOW).to_edge(DOWN)  # fmt: skip
        self.play(Transform(caption, verdict))
        self.wait(2.5)
        self.play(FadeOut(*self.mobjects))
