"""BERT's masking recipe: pick 15% of the tokens; turn 80% of those into [MASK], 10% into a
random word, keep 10%; predict the original at every picked position, and score only those.

Render: make videos paper=bert scene=masking-recipe
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
    YELLOW,
    FadeIn,
    FadeOut,
    GrowFromEdge,
    Indicate,
    LaggedStart,
    Rectangle,
    RoundedRectangle,
    Scene,
    Text,
    Transform,
    VGroup,
)

ROWS = [
    "the old farmer walked his dog to the river",
    "and back every morning before the sun came up",
]
#: (row, column) of the picked words, what happens to each, and the colour of that outcome.
PICKED = [
    ((0, 5), "[MASK]", YELLOW, "80%: [MASK]"),
    ((1, 7), "apple", RED, "10%: a random word"),
    ((0, 8), "river", GREEN, "10%: left as it was"),
]


def word_box(word: str) -> VGroup:
    label = Text(word, font_size=22)
    box = RoundedRectangle(corner_radius=0.06, width=max(label.width + 0.25, 0.9), height=0.55)
    return VGroup(box.set_stroke(GREY_B, 1.5), label)


class MaskingRecipe(Scene):
    def construct(self) -> None:
        title = Text("How BERT hides words", font_size=36).to_edge(UP)
        rows = VGroup(
            *(VGroup(*(word_box(w) for w in row.split())).arrange(buff=0.12) for row in ROWS)
        ).arrange(DOWN, buff=0.3)
        rows.next_to(title, DOWN, buff=0.5)
        self.play(FadeIn(title), LaggedStart(*(FadeIn(r) for r in rows), lag_ratio=0.3))

        # Step 1: pick 15% of the tokens, here 3 of 18.
        step = Text("1. pick 15% of the words at random (3 of these 18)", font_size=24)
        step.next_to(rows, DOWN, buff=0.4)
        boxes = [rows[r][c] for (r, c), *_ in PICKED]
        self.play(FadeIn(step))
        self.play(*(b[0].animate.set_stroke(BLUE, 4) for b in boxes))
        self.wait(1)

        # Step 2: three different fates for the picked words.
        new_step = Text("2. most become [MASK], but not all", font_size=24).move_to(step)
        self.play(Transform(step, new_step))
        legend = VGroup()
        for box, (_, word, color, text) in zip(boxes, PICKED):
            replacement = Text(word, font_size=22, color=color).move_to(box[1])
            replacement.scale_to_fit_width(min(replacement.width, box[0].width - 0.12))
            self.play(
                Transform(box[1], replacement), box[0].animate.set_stroke(color, 4), run_time=0.7
            )
            legend.add(Text(text, font_size=22, color=color))
        bars = VGroup(
            *(
                Rectangle(width=w, height=0.35).set_stroke(width=0).set_fill(c, 0.85)
                for w, c in ((6.4, YELLOW), (0.8, RED), (0.8, GREEN))
            )
        ).arrange(buff=0)
        bars.next_to(step, DOWN, buff=0.35)
        legend.arrange(buff=0.6).next_to(bars, DOWN, buff=0.15)
        self.play(GrowFromEdge(bars, LEFT), FadeIn(legend))
        self.wait(1.5)

        why = Text(
            "[MASK] never appears in real text, so the model can't trust any word blindly",
            font_size=20,
            color=GREY_B,
        ).next_to(legend, DOWN, buff=0.25)
        self.play(FadeIn(why))
        self.wait(2)

        # Step 3: predict the original word at each picked position; nothing else is scored.
        self.play(FadeOut(bars, legend, why))
        new_step = Text(
            "3. predict the original word at those 3 places, and only there", font_size=24
        )
        new_step.move_to(step)
        self.play(Transform(step, new_step))
        predictions = VGroup(
            *(
                VGroup(
                    Text(shown, font_size=24, color=color),
                    Text("→", font_size=24),
                    Text(original, font_size=24, color=BLUE),
                ).arrange(buff=0.2)
                for (_, shown, color, _), original in zip(PICKED, ("dog", "came", "river"))
            )
        ).arrange(buff=1.0)
        predictions.next_to(step, DOWN, buff=0.6)
        for box, prediction in zip(boxes, predictions):
            self.play(Indicate(box, color=ORANGE), FadeIn(prediction, shift=UP * 0.2), run_time=0.7)
        note = Text(
            "the other 15 words are given, not asked: they cost nothing", font_size=20, color=GREY_B
        )
        note.next_to(predictions, DOWN, buff=0.3)
        self.play(FadeIn(note))
        loss = Text("loss = cross-entropy at the picked words", font_size=24, color=ORANGE)
        loss.to_edge(DOWN)
        self.play(FadeIn(loss))
        self.wait(2)
        self.play(FadeOut(*self.mobjects))
