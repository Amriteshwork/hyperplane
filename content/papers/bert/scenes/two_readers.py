"""Two readers: a left-to-right model meets a gap with only the left in view; BERT sees the
whole sentence, finds "barked" on the right and fills in "dog".

Render: make videos paper=bert scene=two-readers
"""

from manim import (
    BLUE,
    DOWN,
    GREEN,
    GREY_B,
    GREY_D,
    ORANGE,
    UP,
    YELLOW,
    Create,
    CurvedArrow,
    FadeIn,
    FadeOut,
    Indicate,
    LaggedStart,
    Rectangle,
    RoundedRectangle,
    Scene,
    Text,
    Transform,
    VGroup,
)

WORDS = ["The", "____", "barked", "at", "the", "postman"]
GAP = 1


def word_box(word: str, color: str = GREY_B) -> VGroup:
    label = Text(word, font_size=28)
    box = RoundedRectangle(corner_radius=0.08, width=max(label.width + 0.35, 1.2), height=0.7)
    return VGroup(box.set_stroke(color, 2), label)


class TwoReaders(Scene):
    def construct(self) -> None:
        title = Text("Fill in the gap", font_size=36).to_edge(UP)
        sentence = VGroup(*(word_box(w) for w in WORDS)).arrange(buff=0.25).shift(UP * 0.6)
        self.play(FadeIn(title), LaggedStart(*(FadeIn(w) for w in sentence), lag_ratio=0.1))
        self.wait(0.5)

        # Reader 1: left to right, so everything after the gap is behind a curtain.
        label = Text("GPT reads left to right", font_size=28, color=ORANGE).next_to(
            sentence, DOWN, 0.6
        )
        right = VGroup(*sentence[GAP + 1 :])
        curtain = Rectangle(width=right.width + 0.3, height=right.height + 0.3)
        curtain.set_stroke(width=0).set_fill(GREY_D, 0.95).move_to(right)
        hidden = Text("not seen yet", font_size=22, color=GREY_B).move_to(curtain)
        self.play(FadeIn(label))
        for box in sentence[: GAP + 1]:
            self.play(Indicate(box, color=ORANGE), run_time=0.5)
        self.play(FadeIn(curtain), FadeIn(hidden))
        guesses = Text("cat?  man?  bus?  wind?", font_size=26, color=ORANGE)
        guesses.next_to(sentence[GAP], DOWN, buff=1.3)
        self.play(FadeIn(guesses, shift=UP * 0.2))
        verdict = Text("only “The” to go on: a guess", font_size=22, color=GREY_B)
        verdict.next_to(guesses, DOWN, buff=0.2)
        self.play(FadeIn(verdict))
        self.wait(1.5)

        # Reader 2: BERT sees every word at once, the gap included.
        new_label = Text("BERT reads the whole sentence at once", font_size=28, color=BLUE)
        new_label.move_to(label)
        self.play(
            FadeOut(curtain, hidden, guesses, verdict),
            Transform(label, new_label),
        )
        arrows = VGroup(
            *(
                # Counter-clockwise arcs bulge upwards when they run right to left.
                CurvedArrow(
                    box.get_top() + UP * 0.05,
                    sentence[GAP].get_top() + UP * 0.05,
                    angle=1.4 if i > GAP else -1.4,
                    tip_length=0.15,
                ).set_stroke(BLUE, 2 + 3 * (i == 2), opacity=0.4 + 0.6 * (i == 2))
                for i, box in enumerate(sentence)
                if i != GAP
            )
        )
        self.play(LaggedStart(*(Create(a) for a in arrows), lag_ratio=0.1))
        self.play(Indicate(sentence[2], color=YELLOW, scale_factor=1.2))
        clue = Text("“barked” is the clue, and it's on the right", font_size=24, color=YELLOW)
        clue.next_to(label, DOWN, buff=0.3)
        self.play(FadeIn(clue))
        answer = Text("dog", font_size=28, color=GREEN).move_to(sentence[GAP][1])
        self.play(
            Transform(sentence[GAP][1], answer), sentence[GAP][0].animate.set_stroke(GREEN, 3)
        )
        self.wait(1)

        summary = Text("Same blocks. GPT has a causal mask; BERT doesn't.", font_size=26)
        summary.to_edge(DOWN)
        self.play(FadeOut(arrows), FadeIn(summary))
        self.wait(2)
        self.play(FadeOut(*self.mobjects))
