"""Votes train a reward model, the reward reshapes the policy, and pushing too far picks the worst reply.

Render: make videos concept=rlhf scene=votes-to-policy
"""

from manim import (
    BLUE,
    DOWN,
    GREEN,
    GREY_B,
    LEFT,
    ORANGE,
    RED,
    UP,
    FadeIn,
    FadeOut,
    Rectangle,
    RoundedRectangle,
    Scene,
    Text,
    Transform,
    VGroup,
)

REPLIES = ["maybe figs.", "no idea.", "jam.", "jam, thanks.", "jam, thanks for asking.", "JAM! thanks thanks..."]
#: The exact best policies at three values of β, from this page's NumPy code.
POLICIES = {
    "the base model": [0.25, 0.18, 0.40, 0.10, 0.05, 0.02],
    "a tight leash": [0.06, 0.05, 0.38, 0.23, 0.16, 0.12],
    "a loose leash": [0.00, 0.00, 0.00, 0.00, 0.00, 1.00],
}
QUALITY = {"the base model": "+1.29", "a tight leash": "+1.64", "a loose leash": "-3.00"}
COLORS = [GREY_B, GREY_B, BLUE, BLUE, GREEN, RED]
WIDE = 5.0


def bars(policy: list[float]) -> VGroup:
    rows = VGroup()
    for i, (p, reply) in enumerate(zip(policy, REPLIES)):
        y = 1.3 - i * 0.55
        label = Text(reply, font_size=20).move_to([-3.6, y, 0])
        bar = Rectangle(width=max(0.02, p * WIDE), height=0.34).set_stroke(width=0)
        bar.set_fill(COLORS[i], 0.85)
        bar.move_to([-1.0, y, 0], aligned_edge=LEFT)
        share = Text(f"{p:.0%}", font_size=18, color=GREY_B).next_to(bar, buff=0.1)
        rows.add(VGroup(label, bar, share))
    return rows


class VotesToPolicy(Scene):
    def construct(self) -> None:
        title = Text("Step 1: which reply is better?", font_size=34).to_edge(UP)
        self.play(FadeIn(title))
        box = RoundedRectangle(corner_radius=0.1, width=3.8, height=1)
        a = VGroup(box.copy().set_stroke(GREEN, 3), Text("jam, thanks.", font_size=26)).move_to([-2.4, 0.6, 0])
        b = VGroup(box.copy().set_stroke(GREY_B, 2), Text("no idea.", font_size=26)).move_to([2.4, 0.6, 0])
        self.play(FadeIn(a), FadeIn(b))
        vote = Text("A person votes: left", font_size=24, color=GREEN).move_to([0, -0.5, 0])
        self.play(FadeIn(vote))
        formula = Text("Train a reward model so that  P(A beats B) = σ( r(A) − r(B) )", font_size=26)
        formula.move_to([0, -1.6, 0])
        self.play(FadeIn(formula))
        self.wait(2.5)
        self.play(FadeOut(a), FadeOut(b), FadeOut(vote), FadeOut(formula))

        title2 = Text("Step 2: shift the model towards high scores", font_size=34).to_edge(UP)
        self.play(Transform(title, title2))
        current = bars(POLICIES["the base model"])
        caption = Text("the base model   ·   real quality +1.29", font_size=26, color=GREY_B)
        caption.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(current), FadeIn(caption))
        self.wait(2)
        for name in ("a tight leash", "a loose leash"):
            if name == "a loose leash":
                title3 = Text("Step 3: push too far, and quality collapses", font_size=34).to_edge(UP)
                self.play(Transform(title, title3))
            color = GREEN if name == "a tight leash" else RED
            text = Text(f"{name}   ·   real quality {QUALITY[name]}", font_size=26, color=color)
            text.to_edge(DOWN, buff=0.4)
            self.play(Transform(current, bars(POLICIES[name])), Transform(caption, text), run_time=2)
            self.wait(2.5)
        moral = Text(
            "It scores highest on the reward model, and nobody ever rated it.", font_size=24, color=ORANGE
        ).next_to(caption, UP, buff=0.15)
        self.play(FadeIn(moral, shift=DOWN * 0.1))
        self.wait(3)
