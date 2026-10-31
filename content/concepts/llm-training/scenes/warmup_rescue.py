"""One model, one peak learning rate: starting at full speed breaks it, and warming up doesn't.

Render: make videos concept=llm-training scene=warmup-rescue
"""

import math

from manim import (
    BLUE,
    DOWN,
    GREEN,
    GREY_B,
    ORANGE,
    UP,
    Axes,
    Create,
    FadeIn,
    Scene,
    Text,
)

PEAK = 0.1
WARMUP = 40
TOTAL = 200


def scheduled(step: float) -> float:
    if step < WARMUP:
        return PEAK * (step + 1) / WARMUP
    return PEAK * 0.5 * (1 + math.cos(math.pi * (step - WARMUP) / (TOTAL - WARMUP)))


class WarmupRescue(Scene):
    def construct(self) -> None:
        title = Text("The same peak, reached gradually", font_size=34).to_edge(UP)
        self.play(FadeIn(title))
        axes = Axes(
            x_range=[0, TOTAL, 50],
            y_range=[0, 0.12, 0.04],
            x_length=8,
            y_length=3.6,
            axis_config={"color": GREY_B, "include_numbers": False},
        ).shift(DOWN * 0.1 + [-1.8, 0, 0])
        x_label = Text("training step", font_size=20, color=GREY_B).next_to(axes.x_axis, DOWN, buff=0.15)
        y_label = Text("learning rate", font_size=20, color=GREY_B).next_to(axes.y_axis, UP, buff=0.1)
        self.play(Create(axes), FadeIn(x_label), FadeIn(y_label))

        flat = axes.plot(lambda _: PEAK, x_range=[0, TOTAL], color=ORANGE)
        self.play(Create(flat), run_time=1.5)
        wild = Text("constant 0.1\nvalidation loss 1.58", font_size=24, color=ORANGE, line_spacing=1)
        wild.move_to([4.4, 1.3, 0])
        self.play(FadeIn(wild))
        note = Text("Big steps on random weights: it breaks, and stays broken", font_size=24, color=ORANGE)
        note.to_edge(DOWN, buff=0.3)
        self.play(FadeIn(note))
        self.wait(2)

        curve = axes.plot(scheduled, x_range=[0, TOTAL, 1], color=BLUE)
        self.play(Create(curve), run_time=2.5)
        good = Text("warmup + cosine\nvalidation loss 0.31", font_size=24, color=GREEN, line_spacing=1)
        good.move_to([4.4, -0.4, 0])
        self.play(FadeIn(good))
        moral = Text("Measured on this page's tiny GPT, from one seed", font_size=24, color=GREEN)
        moral.to_edge(DOWN, buff=0.3)
        self.play(note.animate.become(moral))
        self.wait(3)
