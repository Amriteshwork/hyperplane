"""Sinusoidal positional encoding as a row of clocks turning at different speeds.

Render: make videos paper=attention-is-all-you-need scene=positional-clocks
"""

import numpy as np
from manim import (
    BLUE,
    DOWN,
    GREY_B,
    UP,
    YELLOW,
    Circle,
    FadeIn,
    FadeOut,
    Line,
    Scene,
    Text,
    Transform,
    ValueTracker,
    VGroup,
    always_redraw,
    linear,
)

#: Radians each clock's hand turns per position; each pair of dimensions is one clock.
SPEEDS = [1.0, 0.5, 0.25, 0.12, 0.06, 0.03]
RADIUS = 0.75


def hand_angle(position: float, speed: float) -> float:
    # sin and cos of position × speed are the clock's two coordinates; 12 o'clock is position 0.
    return np.pi / 2 - position * speed


class PositionalClocks(Scene):
    def construct(self) -> None:
        title = Text("Every position gets a fingerprint", font_size=36).to_edge(UP)
        faces = VGroup(*(Circle(RADIUS).set_stroke(GREY_B, 2) for _ in SPEEDS)).arrange(buff=0.45)
        speeds = VGroup(
            *(
                Text(label, font_size=18, color=GREY_B).next_to(f, DOWN)
                for f, label in zip(faces, ["fastest", "", "", "", "", "slowest"])
            )
        )
        pos = ValueTracker(0)

        def hands() -> VGroup:
            return VGroup(
                *(
                    Line(
                        f.get_center(),
                        f.get_center()
                        + 0.9 * RADIUS * np.array([np.cos(a), np.sin(a), 0]),
                    ).set_stroke(YELLOW if i < 2 else BLUE, 5)
                    for i, f in enumerate(faces)
                    for a in [hand_angle(pos.get_value(), SPEEDS[i])]
                )
            )

        clock_hands = always_redraw(hands)
        readout = always_redraw(
            lambda: Text(f"position {pos.get_value():.0f}", font_size=30).next_to(faces, UP, buff=0.6)
        )
        self.play(FadeIn(title), FadeIn(faces), FadeIn(speeds))
        self.add(clock_hands, readout)
        self.wait(0.5)

        caption = Text("Fast clocks tell neighbouring words apart…", font_size=26).to_edge(DOWN)
        self.play(FadeIn(caption))
        self.play(pos.animate.set_value(6), run_time=4, rate_func=linear)
        self.wait(0.5)
        self.play(
            Transform(caption, Text("…slow clocks tell far-apart words apart.", font_size=26).to_edge(DOWN))
        )
        self.play(pos.animate.set_value(60), run_time=5, rate_func=linear)
        self.wait(0.5)

        # Moving k positions turns every hand by the same amount, wherever you start.
        rule = Text(
            "+3 positions = the same turn of every hand, from any start",
            font_size=26,
            color=YELLOW,
        ).to_edge(DOWN)
        self.play(Transform(caption, rule), pos.animate.set_value(10))
        for start in (10, 40):
            self.play(pos.animate.set_value(start), run_time=0.8)
            self.wait(0.3)
            self.play(pos.animate.set_value(start + 3), run_time=1.5)
            self.wait(0.6)
        note = Text("So “look 3 words back” is one rule the model can learn once.", font_size=22)
        note.next_to(caption, UP)
        self.play(FadeIn(note))
        self.wait(2)
        self.play(FadeOut(*self.mobjects))
