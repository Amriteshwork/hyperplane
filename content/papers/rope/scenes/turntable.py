"""RoPE in 2D: q turns by m·θ, k by n·θ; shift both and the angle between them never changes.

Render: make videos paper=rope scene=turntable
"""

import numpy as np
from manim import (
    BLUE,
    DOWN,
    GREY_B,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    WHITE,
    YELLOW,
    Arc,
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
)

THETA = np.deg2rad(20)  # turn per position
RADIUS = 2.4
Q_ANGLE = np.deg2rad(15)  # where q and k point at position 0
K_ANGLE = np.deg2rad(25)


def tip(angle: float) -> np.ndarray:
    return RADIUS * np.array([np.cos(angle), np.sin(angle), 0])


class Turntable(Scene):
    def construct(self) -> None:
        title = Text("Rotate each vector by its position", font_size=34).to_edge(UP)
        dial = Circle(RADIUS).set_stroke(GREY_B, 1.5).shift(LEFT * 2.2 + DOWN * 0.4)
        centre = dial.get_center()
        m, n = ValueTracker(0), ValueTracker(0)

        def arrow(angle: float, colour: str) -> Arrow:
            return Arrow(centre, centre + tip(angle), buff=0, color=colour, stroke_width=6)

        q = always_redraw(lambda: arrow(Q_ANGLE + m.get_value() * THETA, YELLOW))
        k = always_redraw(lambda: arrow(K_ANGLE + n.get_value() * THETA, BLUE))

        def gap() -> Arc:
            a, b = Q_ANGLE + m.get_value() * THETA, K_ANGLE + n.get_value() * THETA
            return Arc(0.7, start_angle=a, angle=b - a, arc_center=centre, color=WHITE)

        arc = always_redraw(gap)

        def readout() -> VGroup:
            mv, nv = m.get_value(), n.get_value()
            gap_deg = np.rad2deg(K_ANGLE - Q_ANGLE + (nv - mv) * THETA)
            score = np.cos(K_ANGLE - Q_ANGLE + (nv - mv) * THETA)
            lines = VGroup(
                Text(f"q at position m = {mv:.0f}", font_size=26, color=YELLOW),
                Text(f"k at position n = {nv:.0f}", font_size=26, color=BLUE),
                Text(f"n − m = {nv - mv:.0f}", font_size=26),
                Text(f"angle between: {gap_deg:.0f}°", font_size=26),
                Text(f"q · k = {score:.3f}", font_size=30, color=WHITE),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
            return lines.next_to(dial, RIGHT, buff=1.0)

        info = always_redraw(readout)
        self.play(FadeIn(title), FadeIn(dial))
        self.add(q, k, arc, info)
        self.wait(1)

        caption = Text("Move the key 3 positions: it turns 3 × 20° = 60°", font_size=24).to_edge(
            DOWN
        )
        self.play(FadeIn(caption))
        self.play(n.animate.set_value(3), run_time=2)
        self.wait(1)

        rule = Text("Now shift BOTH by +5: they turn together", font_size=24, color=YELLOW)
        self.play(Transform(caption, rule.to_edge(DOWN)))
        self.play(m.animate.set_value(5), n.animate.set_value(8), run_time=3)
        self.wait(0.5)
        self.play(m.animate.set_value(12), n.animate.set_value(15), run_time=3)
        self.wait(1)

        moral = Text(
            "The angle, and so q · k, only knows the gap n − m",
            font_size=26,
            color=YELLOW,
        ).to_edge(DOWN)
        self.play(Transform(caption, moral))
        self.wait(2.5)
        self.play(FadeOut(*self.mobjects), run_time=0.8)
        self.wait(0.2)
        # Leave on a clean frame so the YouTube thumbnail isn't mid-motion.
        self.add(Text("Like two people on a turntable", font_size=34).move_to(ORIGIN))
        self.wait(1)
