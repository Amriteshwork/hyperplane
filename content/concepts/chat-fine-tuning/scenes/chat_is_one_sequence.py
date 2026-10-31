"""A chat flattens into one token sequence, and only the assistant's tokens count towards the loss.

Render: make videos concept=chat-fine-tuning scene=chat-is-one-sequence
"""

from manim import (
    BLUE,
    DOWN,
    GREEN,
    GREY_B,
    LEFT,
    ORANGE,
    RIGHT,
    UP,
    FadeIn,
    FadeOut,
    LaggedStart,
    RoundedRectangle,
    Scene,
    Text,
    Transform,
    VGroup,
)

#: Token, and who's speaking.
TOKENS = [
    ("<user>", "m"), ("what", "u"), ("does", "u"), ("zoe", "u"), ("eat", "u"), ("?", "u"),
    ("<assistant>", "m"), ("jam", "a"), (".", "a"), ("<stop>", "a"),
]  # fmt: skip
COLORS = {"m": GREY_B, "u": BLUE, "a": GREEN}


def chip(text: str, color: str) -> VGroup:
    label = Text(text, font_size=20)
    box = RoundedRectangle(corner_radius=0.08, width=max(label.width + 0.22, 0.72), height=0.55)
    box.set_stroke(color, 2).set_fill(color, 0.15)
    return VGroup(box, label.move_to(box))


class ChatIsOneSequence(Scene):
    def construct(self) -> None:
        title = Text("What you see", font_size=34).to_edge(UP)
        self.play(FadeIn(title))
        user = chip("what does zoe eat?", BLUE).scale(1.3).move_to([2.6, 1.1, 0])
        assistant = chip("jam.", GREEN).scale(1.3).move_to([-3.6, -0.1, 0])
        self.play(FadeIn(user, shift=LEFT * 0.3))
        self.wait(0.5)
        self.play(FadeIn(assistant, shift=RIGHT * 0.3))
        self.wait(1.5)

        title2 = Text("What the model sees: one sequence", font_size=34).to_edge(UP)
        chips = VGroup(*(chip(t, COLORS[s]) for t, s in TOKENS)).arrange(buff=0.08).move_to([0, 0.6, 0])
        self.play(Transform(title, title2), FadeOut(user), FadeOut(assistant))
        self.play(LaggedStart(*(FadeIn(c, shift=DOWN * 0.2) for c in chips), lag_ratio=0.1))
        note = Text("It still only predicts the next token", font_size=26, color=GREY_B)
        note.move_to([0, -0.6, 0])
        self.play(FadeIn(note))
        self.wait(2)

        title3 = Text("Which predictions count?", font_size=34).to_edge(UP)
        self.play(Transform(title, title3), FadeOut(note))
        marks = VGroup()
        for c, (_, speaker) in zip(chips, TOKENS):
            taught = speaker == "a"
            mark = Text("loss" if taught else "-100", font_size=18, color=GREEN if taught else GREY_B)
            marks.add(mark.next_to(c, DOWN, buff=0.15))
        self.play(LaggedStart(*(FadeIn(m) for m in marks), lag_ratio=0.1))
        self.play(*(c.animate.set_opacity(0.3) for c, (_, s) in zip(chips, TOKENS) if s != "a"))
        moral = VGroup(
            Text("The user's words cost nothing, so it never learns to write them.", font_size=24),
            Text("The stop token counts. That's how it learns to finish.", font_size=24, color=ORANGE),
        ).arrange(DOWN, buff=0.2)
        moral.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(moral))
        self.wait(3.5)
