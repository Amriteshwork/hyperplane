"""A slow clock has only seen a slice of its circle: extrapolation runs off the slice, Position
Interpolation stays on it, but squeezes the fast clock until neighbours nearly overlap.

Render: make videos paper=yarn scene=seen-angles
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
    AnnularSector,
    Arrow,
    Circle,
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

RADIUS = 1.8
L = 4096  # trained length
S = 4  # stretch
SEEN = 2 * np.pi / 3  # the slow clock turns 120° across the training length
SLOW = SEEN / L  # its speed in radians per token
START = np.pi / 2  # 12 o'clock is position 0; hands turn clockwise


def hand_tip(centre: np.ndarray, angle: float, length: float = 0.92) -> np.ndarray:
    a = START - angle
    return centre + length * RADIUS * np.array([np.cos(a), np.sin(a), 0])


class SeenAngles(Scene):
    def construct(self) -> None:
        title = Text("A slow clock has only seen part of its circle", font_size=32).to_edge(UP)
        centre = DOWN * 0.3 + LEFT * 2.4
        face = Circle(RADIUS).set_stroke(GREY_B, 2).move_to(centre)
        seen = AnnularSector(
            inner_radius=0,
            outer_radius=RADIUS,
            angle=-SEEN,
            start_angle=START,
            color=BLUE,
            fill_opacity=0.25,
        ).move_arc_center_to(centre)
        seen_label = Text("angles seen\nin training", font_size=22, color=BLUE).next_to(
            face, RIGHT, buff=0.3
        )
        position = ValueTracker(0)
        mode = {"squeeze": 1.0}

        def hand() -> Arrow:
            angle = position.get_value() / mode["squeeze"] * SLOW
            colour = RED if angle > SEEN + 1e-6 else YELLOW
            return Arrow(centre, hand_tip(centre, angle), buff=0, color=colour, stroke_width=7)

        def readout() -> Text:
            squeeze = mode["squeeze"]
            label = f"token {position.get_value():,.0f}"
            if squeeze > 1:
                label += f"  →  treated as {position.get_value() / squeeze:,.0f}"
            return Text(label, font_size=26).move_to(RIGHT * 3 + UP * 1.2)

        self.play(FadeIn(title), FadeIn(face), FadeIn(seen), FadeIn(seen_label))
        self.add(always_redraw(hand), always_redraw(readout))
        caption = Text(f"Training: tokens 0 to {L:,}", font_size=26).to_edge(DOWN)
        self.play(FadeIn(caption))
        self.play(position.animate.set_value(L), run_time=3, rate_func=linear)
        self.wait(0.5)

        self.play(
            Transform(
                caption,
                Text(
                    f"Extrapolate to {S * L:,} tokens: the hand leaves what it knows",
                    font_size=26,
                    color=RED,
                ).to_edge(DOWN),
            )
        )
        self.play(position.animate.set_value(S * L), run_time=4, rate_func=linear)
        self.wait(1)
        self.play(position.animate.set_value(0), run_time=1)

        mode["squeeze"] = S
        self.play(
            Transform(
                caption,
                Text(
                    f"Position Interpolation: divide every position by {S}",
                    font_size=26,
                    color=YELLOW,
                ).to_edge(DOWN),
            )
        )
        self.play(position.animate.set_value(S * L), run_time=4, rate_func=linear)
        self.wait(1)
        self.play(FadeOut(*self.mobjects))

        # The price: on the fastest clock (1 rad per token) neighbours used to sit 1 radian apart.
        title2 = Text("…but the fastest clock pays for it", font_size=32).to_edge(UP)
        faces = VGroup(*(Circle(1.4).set_stroke(GREY_B, 2) for _ in range(2))).arrange(buff=2.5)
        faces.shift(DOWN * 0.2)
        heads = [
            Text("before: 1 rad between neighbours", font_size=22),
            Text("after ÷16: 0.06 rad", font_size=22, color=RED),
        ]
        for f, h in zip(faces, heads, strict=True):
            h.next_to(f, UP)
        hands = VGroup()
        for f, step in zip(faces, [1.0, 1 / 16], strict=True):
            c = f.get_center()
            for k, colour in enumerate([YELLOW, BLUE]):
                a = START - (5 + k) * step
                tip = c + 1.25 * np.array([np.cos(a), np.sin(a), 0])
                hands.add(Arrow(c, tip, buff=0, color=colour, stroke_width=6))
        legend = Text("yellow: word 5    blue: word 6", font_size=22, color=GREY_B).to_edge(DOWN)
        self.play(FadeIn(title2), FadeIn(faces), *(FadeIn(h) for h in heads))
        self.play(FadeIn(hands), FadeIn(legend))
        self.wait(1.5)
        moral = Text(
            "Neighbours nearly overlap: the model loses its sense of “right next to me”",
            font_size=24,
            color=YELLOW,
        ).to_edge(DOWN)
        self.play(Transform(legend, moral))
        self.wait(2.5)
        self.play(FadeOut(*self.mobjects))
