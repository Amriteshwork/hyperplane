"""Text becomes tokens, tokens become ids, ids select rows of a table: the whole way into a model.

These are GPT-2's real tokens, ids and embedding size.
Render: make videos concept=tokenization scene=text-to-vector
"""

from manim import (
    BLUE,
    DOWN,
    GREEN,
    GREY_B,
    ORANGE,
    PURPLE,
    UP,
    Arrow,
    FadeIn,
    GrowArrow,
    LaggedStart,
    Rectangle,
    RoundedRectangle,
    Scene,
    Text,
    VGroup,
)

#: From tiktoken's `gpt2` encoding.
TOKENS = [("un", 403), ("bel", 6667), ("iev", 11203), ("able", 540)]
COLORS = [BLUE, ORANGE, GREEN, PURPLE]
#: The size of the first 10 numbers in each token's real row of GPT-2's embedding table, scaled to fit.
SHADES = [[0.01, 0.19, 0.07, 0.83, 0.12, 0.19, 0.86, 0.05, 0.21, 0.28], [0.24, 0.6, 0.25, 0.42, 0.4, 0.08, 0.86, 0.26, 0.68, 0.27], [0.29, 0.49, 0.48, 0.35, 0.49, 0.33, 1.0, 0.24, 0.32, 0.44], [0.05, 0.52, 0.0, 0.65, 0.28, 0.62, 0.82, 0.01, 0.45, 0.3]]
GAP = 3.0


class TextToVector(Scene):
    def construct(self) -> None:
        title = Text("From text to the numbers a model reads", font_size=34).to_edge(UP)
        self.play(FadeIn(title))
        text = Text('"unbelievable"', font_size=38).move_to([0, 2.2, 0])
        self.play(FadeIn(text))
        self.wait(1)

        xs = [(i - 1.5) * GAP for i in range(4)]
        chips = VGroup()
        for (token, _), x, color in zip(TOKENS, xs, COLORS):
            box = RoundedRectangle(corner_radius=0.08, width=1.9, height=0.7).move_to([x, 1.0, 0])
            box.set_stroke(color, 2).set_fill(color, 0.2)
            chips.add(VGroup(box, Text(token, font_size=28).move_to(box)))
        step1 = Text("1. Split it into tokens. Rarer words break into familiar pieces.", font_size=22, color=GREY_B)
        step1.to_edge(DOWN, buff=0.35)
        self.play(FadeIn(step1), LaggedStart(*(FadeIn(c, shift=DOWN * 0.3) for c in chips), lag_ratio=0.3))
        self.wait(2)

        ids = VGroup(
            *(
                Text(f"id {i}", font_size=26, color=color).move_to([x, 0, 0])
                for (_, i), x, color in zip(TOKENS, xs, COLORS)
            )
        )
        arrows = [Arrow(c.get_bottom(), i.get_top(), buff=0.05, color=GREY_B) for c, i in zip(chips, ids)]
        step2 = Text("2. Each token has a number. It's a name, not a quantity.", font_size=22, color=GREY_B)
        step2.to_edge(DOWN, buff=0.35)
        self.play(step1.animate.become(step2), *(GrowArrow(a) for a in arrows), FadeIn(ids))
        self.wait(2)

        rows = VGroup()
        for shades, x, color in zip(SHADES, xs, COLORS):
            cells = VGroup(
                *(
                    Rectangle(width=0.24, height=0.55).set_stroke(width=0).set_fill(color, s)
                    for s in shades
                )
            ).arrange(buff=0.02)
            rows.add(cells.move_to([x, -1.2, 0]))
        arrows2 = [Arrow(i.get_bottom(), r.get_top(), buff=0.05, color=GREY_B) for i, r in zip(ids, rows)]
        step3 = Text("3. The id picks a row of a table: 768 learned numbers in GPT-2.", font_size=22, color=GREY_B)
        step3.to_edge(DOWN, buff=0.35)
        self.play(step1.animate.become(step3), *(GrowArrow(a) for a in arrows2), FadeIn(rows))
        self.wait(3)
