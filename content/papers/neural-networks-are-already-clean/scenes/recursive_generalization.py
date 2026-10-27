"""Recursive generalization: a neuron grows outward in small, sensible steps, like a branch.

Render: make videos paper=neural-networks-are-already-clean scene=recursive-generalization
"""

from manim import (
    BLUE,
    DOWN,
    GREEN,
    GREY_B,
    LEFT,
    RIGHT,
    UP,
    YELLOW,
    Arrow,
    FadeIn,
    FadeOut,
    GrowArrow,
    RoundedRectangle,
    Scene,
    Text,
    Transform,
    VGroup,
)

#: Each step: (label, parent label or None, position, colour). From the essay's "iy" neuron.
STEPS = [
    ("“iy”", None, [-5.0, 0.0, 0], YELLOW),
    ("Miyazaki", "“iy”", [-1.5, 1.5, 0], YELLOW),
    ("diy", "“iy”", [-1.5, -1.0, 0], BLUE),
    ("do it yourself", "diy", [2.8, -1.0, 0], GREEN),
]


def node(label: str, color: str) -> VGroup:
    text = Text(label, font_size=24, color=color)
    box = RoundedRectangle(corner_radius=0.15, width=text.width + 0.5, height=text.height + 0.4)
    return VGroup(box.set_stroke(color, 2), text)


class RecursiveGeneralization(Scene):
    def construct(self) -> None:
        title = Text("A neuron that grows like a branch", font_size=36).to_edge(UP)
        self.play(FadeIn(title))
        nodes: dict[str, VGroup] = {}
        caption = Text("It starts on the letters “iy”…", font_size=24).to_edge(DOWN)

        for i, (label, parent, pos, color) in enumerate(STEPS):
            nodes[label] = node(label, color).move_to(pos)
            if parent is None:
                self.play(FadeIn(nodes[label]), FadeIn(caption))
                continue
            arrow = Arrow(nodes[parent].get_right(), nodes[label].get_left(), buff=0.1)
            self.play(GrowArrow(arrow.set_color(GREY_B)), FadeIn(nodes[label], shift=RIGHT * 0.3))
            if i == 1:
                self.play(
                    Transform(
                        caption,
                        Text("…grows to words that contain it…", font_size=24).to_edge(DOWN),
                    )
                )
            elif i == len(STEPS) - 1:
                self.play(Transform(caption, Text("…and to what those words mean.", font_size=24).to_edge(DOWN)))  # fmt: skip
            self.wait(0.6)

        # The punchline: the last node shares nothing with the first.
        far = nodes["do it yourself"]
        no_iy = Text("no “iy” left at all", font_size=22, color=GREEN).next_to(far, DOWN, buff=0.15)
        self.play(FadeIn(no_iy), far.animate.scale(1.1))
        self.wait(1.5)

        start, end = nodes["“iy”"], far
        gap = Text("Start and end look unrelated…", font_size=24).to_edge(DOWN)
        self.play(
            Transform(caption, gap), start.animate.shift(LEFT * 0.2), end.animate.shift(RIGHT * 0.2)
        )
        self.wait(1)
        self.play(
            Transform(
                caption,
                Text("…but every step was a small, sensible move. That's the clue.",
                     font_size=24, color=YELLOW).to_edge(DOWN),
            )
        )  # fmt: skip
        self.wait(1.5)
        note = Text("Watch it grow over training and the mixture has a history.", font_size=22)
        self.play(FadeIn(note.next_to(caption, UP, buff=0.3)))
        self.wait(2.5)
        self.play(FadeOut(*self.mobjects), run_time=0.8)
