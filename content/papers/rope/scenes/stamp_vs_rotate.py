"""Adding a position stamp changes a vector's length and direction; rotating changes direction only.

Render: make videos paper=rope scene=stamp-vs-rotate
"""

import numpy as np
from manim import (
    BLUE,
    DOWN,
    GREY_B,
    LEFT,
    ORANGE,
    RIGHT,
    UP,
    YELLOW,
    Arrow,
    Circle,
    FadeIn,
    FadeOut,
    Scene,
    Text,
    Transform,
    ValueTracker,
    always_redraw,
    linear,
)

THETA = 0.6  # radians per position
WORD = np.array([1.6, 0.6, 0])  # the word's vector, before any position
STAMP = 0.9  # length of the added position vector


def stamp(position: float) -> np.ndarray:
    return STAMP * np.array([np.sin(position * THETA), np.cos(position * THETA), 0])


def rotated(position: float) -> np.ndarray:
    a = position * THETA
    x, y, _ = WORD
    return np.array([x * np.cos(a) - y * np.sin(a), x * np.sin(a) + y * np.cos(a), 0])


class StampVsRotate(Scene):
    def construct(self) -> None:
        title = Text("Two ways to say “I'm at position t”", font_size=34).to_edge(UP)
        left, right = LEFT * 3.4 + DOWN * 0.3, RIGHT * 3.4 + DOWN * 0.3
        heads = [
            Text("Add a stamp", font_size=28, color=ORANGE).move_to(left + UP * 2.4),
            Text("Rotate", font_size=28, color=BLUE).move_to(right + UP * 2.4),
        ]
        ring = Circle(np.linalg.norm(WORD)).set_stroke(GREY_B, 1).move_to(right)
        t = ValueTracker(0)

        word = Arrow(left, left + WORD, buff=0, color=GREY_B, stroke_width=4)
        bump = always_redraw(
            lambda: Arrow(
                left + WORD,
                left + WORD + stamp(t.get_value()),
                buff=0,
                color=ORANGE,
                stroke_width=4,
            )
        )
        added = always_redraw(
            lambda: Arrow(
                left, left + WORD + stamp(t.get_value()), buff=0, color=YELLOW, stroke_width=6
            )
        )
        turned = always_redraw(
            lambda: Arrow(
                right, right + rotated(t.get_value()), buff=0, color=YELLOW, stroke_width=6
            )
        )

        def lengths() -> Text:
            a = np.linalg.norm(WORD + stamp(t.get_value()))
            b = np.linalg.norm(rotated(t.get_value()))
            return (
                Text(
                    f"t = {t.get_value():4.1f}      length {a:.2f}            length {b:.2f}",
                    font_size=24,
                )
                .next_to(heads[0], DOWN, buff=3.9)
                .set_x(0)
            )

        readout = always_redraw(lengths)
        self.play(FadeIn(title), *(FadeIn(h) for h in heads), FadeIn(ring), FadeIn(word))
        self.add(bump, added, turned, readout)
        self.wait(1)
        self.play(t.animate.set_value(10), run_time=8, rate_func=linear)
        self.wait(0.5)

        caption = Text(
            "Adding mixes “where” into “what”. Rotating keeps the meaning's size intact.",
            font_size=24,
            color=YELLOW,
        ).to_edge(DOWN)
        self.play(FadeIn(caption))
        self.wait(2)
        follow = Text(
            "…and two rotated vectors compare by the angle between them: the gap.",
            font_size=24,
            color=YELLOW,
        ).to_edge(DOWN)
        self.play(Transform(caption, follow))
        self.wait(2.5)
        self.play(FadeOut(*self.mobjects))
