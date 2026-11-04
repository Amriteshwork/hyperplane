"""NTK-by-parts on Llama 2's clocks: fast clocks (many turns in training) stay as they are, slow
clocks (under one turn) are slowed by s, so no hand ever leaves the angles it saw in training.

Render: make videos paper=yarn scene=clocks-by-parts
"""

import numpy as np
from manim import (
    BLUE,
    DOWN,
    GREEN,
    GREY_B,
    RED,
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

D, BASE, L, S = 128, 10_000, 4096, 16
ALPHA, BETA = 1, 32
PAIRS = [0, 20, 32, 46, 55, 63]  # a fast, a ramp and four slow-ish clocks of Llama 2
RADIUS = 0.8
START = np.pi / 2


def theta(d: int) -> float:
    return BASE ** (-2 * d / D)


def by_parts(t: float) -> float:
    r = L * t / (2 * np.pi)
    g = float(np.clip((r - ALPHA) / (BETA - ALPHA), 0, 1))
    return (1 - g) * t / S + g * t


class ClocksByParts(Scene):
    def construct(self) -> None:
        title = Text("NTK-by-parts: ask each clock how many turns it made", font_size=30)
        title.to_edge(UP)
        faces = VGroup(*(Circle(RADIUS).set_stroke(GREY_B, 2) for _ in PAIRS)).arrange(buff=0.45)
        faces.shift(UP * 0.4)
        seen = VGroup()
        labels = VGroup()
        rules = VGroup()
        for f, d in zip(faces, PAIRS, strict=True):
            turns = L * theta(d) / (2 * np.pi)
            sweep = min(turns, 1.0) * 2 * np.pi
            seen.add(
                AnnularSector(
                    inner_radius=0,
                    outer_radius=RADIUS,
                    angle=-sweep,
                    start_angle=START,
                    color=BLUE,
                    fill_opacity=0.25,
                ).move_arc_center_to(f.get_center())
            )
            text = f"{turns:.0f} turns" if turns >= 10 else f"{turns:.2g} turns"
            labels.add(Text(text, font_size=18, color=GREY_B).next_to(f, UP, buff=0.15))
            if turns > BETA:
                rule, colour = "leave alone", GREEN
            elif turns < ALPHA:
                rule, colour = "slow ×16", YELLOW
            else:
                rule, colour = "blend", BLUE
            rules.add(Text(rule, font_size=18, color=colour).next_to(f, DOWN, buff=0.2))

        position = ValueTracker(0)
        speeds = {"fn": theta}

        def hands() -> VGroup:
            group = VGroup()
            for f, d in zip(faces, PAIRS, strict=True):
                t = theta(d)
                angle = position.get_value() * speeds["fn"](d)
                outside = t * L < 2 * np.pi and angle > t * L + 1e-6
                a = START - angle
                tip = f.get_center() + 0.9 * RADIUS * np.array([np.cos(a), np.sin(a), 0])
                group.add(
                    Arrow(
                        f.get_center(),
                        tip,
                        buff=0,
                        color=RED if outside else YELLOW,
                        stroke_width=5,
                    )
                )
            return group

        readout = always_redraw(
            lambda: Text(f"token {position.get_value():,.0f}", font_size=26).next_to(
                faces, DOWN, buff=1.0
            )
        )
        self.play(FadeIn(title), FadeIn(faces), FadeIn(seen), FadeIn(labels))
        self.add(always_redraw(hands), readout)
        caption = Text(
            "Shaded: the angles each clock reached in 4,096 training tokens", font_size=24
        ).to_edge(DOWN)
        self.play(FadeIn(caption))
        self.play(position.animate.set_value(L), run_time=3, rate_func=linear)
        self.wait(0.5)

        self.play(
            Transform(
                caption,
                Text(
                    "Read 16× further, unchanged: slow hands run off their slices",
                    font_size=24,
                    color=RED,
                ).to_edge(DOWN),
            ),
            position.animate.set_value(0),
        )
        self.play(position.animate.set_value(S * L), run_time=5, rate_func=linear)
        self.wait(1)

        self.play(
            Transform(
                caption,
                Text(
                    "By parts: fast clocks untouched, slow ones ×16 slower",
                    font_size=24,
                    color=YELLOW,
                ).to_edge(DOWN),
            ),
            FadeIn(rules),
            position.animate.set_value(0),
        )
        speeds["fn"] = lambda d: by_parts(theta(d))
        self.play(position.animate.set_value(S * L), run_time=5, rate_func=linear)
        self.wait(1)
        self.play(
            Transform(
                caption,
                Text(
                    "Every hand stays on angles it knows, and the fast clocks keep their detail",
                    font_size=24,
                    color=YELLOW,
                ).to_edge(DOWN),
            )
        )
        self.wait(2.5)
        self.play(FadeOut(*self.mobjects))
