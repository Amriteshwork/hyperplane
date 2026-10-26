"""Attention as a soft dictionary lookup: score each key, softmax into shares, blend the values.

Render: make videos paper=attention-is-all-you-need scene=qkv-lookup
"""

import numpy as np
from manim import (
    BLUE,
    DOWN,
    GREEN,
    GREY_B,
    LEFT,
    ORANGE,
    RIGHT,
    UP,
    YELLOW,
    FadeIn,
    FadeOut,
    GrowFromEdge,
    LaggedStart,
    ManimColor,
    Rectangle,
    RoundedRectangle,
    Scene,
    Square,
    Text,
    Transform,
    VGroup,
)

#: The words "it" can look at, what each key offers, its value's colour, and the match score.
ROWS = [
    ("animal", "noun · alive · can get tired", BLUE, 4.1),
    ("street", "noun · a place", GREY_B, 1.2),
    ("cross", "verb · movement", GREEN, 0.3),
    ("tired", "adjective · a state", ORANGE, 2.0),
]
BAR_X = -1.8
UNIT = 0.55  # bar width per score point


def softmax(scores: list[float]) -> list[float]:
    e = np.exp(scores)
    return list(e / e.sum())


def bar(width: float, color: ManimColor, row: VGroup) -> Rectangle:
    rect = Rectangle(width=max(width, 0.02), height=0.36).set_stroke(width=0).set_fill(color, 0.85)
    return rect.move_to(row, aligned_edge=LEFT).set_x(BAR_X + rect.width / 2)


class QkvLookup(Scene):
    def construct(self) -> None:
        title = Text("Attention is a soft dictionary lookup", font_size=36).to_edge(UP)
        query = VGroup(
            Text("query of “it”", font_size=24, color=YELLOW),
            Text("“a noun that can get tired”", font_size=22),
        ).arrange(DOWN, buff=0.1)
        query.add_to_back(
            RoundedRectangle(corner_radius=0.1, width=query.width + 0.4, height=query.height + 0.3)
            .set_stroke(YELLOW)
            .move_to(query)
        )
        query.next_to(title, DOWN, buff=0.3)
        self.play(FadeIn(title), FadeIn(query))

        keys = VGroup(
            *(
                VGroup(
                    Text(word, font_size=26, color=color),
                    Text(offer, font_size=18, color=GREY_B),
                ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
                for word, offer, color, _ in ROWS
            )
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        keys.to_edge(LEFT, buff=0.8).shift(DOWN * 0.9)
        keys_label = Text("keys", font_size=22, color=GREY_B).next_to(keys, UP, aligned_edge=LEFT)
        self.play(LaggedStart(*(FadeIn(k) for k in keys), lag_ratio=0.15), FadeIn(keys_label))

        # Step 1: score = how well the query matches each key.
        step = Text("1. score = query · key  (÷ √d keeps it small)", font_size=24)
        step.next_to(query, DOWN, buff=0.25)
        scores = [score for *_, score in ROWS]
        bars = VGroup(*(bar(s * UNIT, YELLOW, k) for s, k in zip(scores, keys)))
        labels = VGroup(
            *(Text(f"{s:.1f}", font_size=22).next_to(b, RIGHT) for s, b in zip(scores, bars))
        )
        self.play(FadeIn(step))
        self.play(*(GrowFromEdge(b, LEFT) for b in bars), FadeIn(labels))
        self.wait(1)

        # Step 2: softmax turns scores into shares that add up to 1.
        weights = softmax(scores)
        new_step = Text("2. softmax → shares that add up to 100%", font_size=24).move_to(step)
        shares = VGroup(*(bar(w * 3.2, YELLOW, k) for w, k in zip(weights, keys)))
        share_labels = VGroup(
            *(Text(f"{w:.0%}", font_size=22).next_to(b, RIGHT) for w, b in zip(weights, shares))
        )
        self.play(
            Transform(step, new_step),
            Transform(bars, shares),
            Transform(labels, share_labels),
        )
        self.wait(1)

        # Step 3: blend the values in those shares.
        new_step = Text("3. blend the values in those shares", font_size=24).move_to(step)
        values = VGroup(
            *(
                Square(0.45).set_stroke(width=0).set_fill(color, 1).next_to(label, RIGHT, buff=0.4)
                for (_, _, color, _), label in zip(ROWS, labels)
            )
        )
        self.play(Transform(step, new_step), FadeIn(values))
        rgb = sum(w * ManimColor(color).to_rgb() for w, (_, _, color, _) in zip(weights, ROWS))
        output = Square(0.9).set_stroke(YELLOW, 3).set_fill(ManimColor.from_rgb(rgb), 1)
        output.move_to([4.5, -0.6, 0])
        pieces = [
            v.copy().animate.scale(max(w, 0.05) * 2).move_to(output).set_opacity(0)
            for v, w in zip(values, weights)
        ]
        self.play(*pieces, FadeIn(output), run_time=1.5)
        caption = Text("new vector for “it”: mostly “animal”", font_size=22, color=YELLOW)
        caption.next_to(output, DOWN)
        self.play(FadeIn(caption))
        self.wait(1)

        summary = Text("softmax(QKᵀ / √d) · V  — every word does this, all at once", font_size=26)
        summary.to_edge(DOWN)
        self.play(FadeIn(summary))
        self.wait(2)
        self.play(FadeOut(*self.mobjects))
