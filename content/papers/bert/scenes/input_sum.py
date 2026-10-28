"""BERT's input: pack two sentences as [CLS] A [SEP] B [SEP], then give every token the sum
of three learned vectors: which token, which sentence, which position.

Render: make videos paper=bert scene=input-sum
"""

from manim import (
    BLUE,
    DOWN,
    GREEN,
    GREY_B,
    LEFT,
    ORANGE,
    PURPLE,
    UP,
    YELLOW,
    FadeIn,
    FadeOut,
    LaggedStart,
    ManimColor,
    RoundedRectangle,
    Scene,
    Square,
    Text,
    VGroup,
)

TOKENS = ["[CLS]", "where", "is", "the", "cat", "[SEP]", "on", "the", "mat", "[SEP]"]
#: Sentence A runs up to and including the first [SEP].
SEGMENTS = [0] * 6 + [1] * 4
SEGMENT_COLORS = [GREEN, ORANGE]
CELL = 1.05


def cell(text: str, color: ManimColor | str, fill: float = 0.25) -> VGroup:
    box = RoundedRectangle(corner_radius=0.06, width=CELL - 0.12, height=0.5)
    box.set_stroke(color, 2).set_fill(color, fill)
    label = Text(text, font_size=16)
    label.scale_to_fit_width(min(label.width, CELL - 0.22))
    return VGroup(box, label.move_to(box))


def row(cells: list[VGroup], y: float) -> VGroup:
    group = VGroup(*cells).arrange(buff=0.12)
    return group.move_to([0.6, y, 0])


class InputSum(Scene):
    def construct(self) -> None:
        title = Text("What BERT actually reads", font_size=34).to_edge(UP)
        self.play(FadeIn(title))

        # One input for two sentences: special tokens frame them.
        tokens = row([cell(t, YELLOW if t.startswith("[") else GREY_B, 0) for t in TOKENS], 2.3)
        caption = Text("A: “where is the cat”   B: “on the mat”", font_size=22, color=GREY_B)
        caption.next_to(tokens, DOWN, buff=0.15)
        self.play(LaggedStart(*(FadeIn(t, shift=DOWN * 0.2) for t in tokens), lag_ratio=0.08))
        self.play(FadeIn(caption))
        notes = VGroup(
            Text(
                "[CLS]: always first; a classifier reads its final vector",
                font_size=20,
                color=YELLOW,
            ),
            Text("[SEP]: marks the end of each sentence", font_size=20, color=YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        notes.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(notes))
        self.wait(2)
        self.play(FadeOut(notes, caption))

        # Three embedding tables, one row each.
        rows = [
            ("token", [cell(t, BLUE) for t in TOKENS], 1.25),
            ("segment", [cell("A" if s == 0 else "B", SEGMENT_COLORS[s]) for s in SEGMENTS], 0.35),
            ("position", [cell(f"P{i}", PURPLE) for i in range(len(TOKENS))], -0.55),
        ]
        drawn = []
        for name, cells, y in rows:
            group = row(cells, y)
            label = Text(name, font_size=20, color=GREY_B).next_to(group, LEFT, buff=0.25)
            plus = Text("+", font_size=26).move_to([0.6, y + 0.45, 0]) if drawn else VGroup()
            self.play(
                FadeIn(label),
                FadeIn(plus),
                LaggedStart(*(FadeIn(c) for c in cells), lag_ratio=0.04),
            )
            drawn.append(VGroup(label, plus, group))
            self.wait(0.6)

        # The sum: one vector per token, carrying all three.
        sums = []
        for i, s in enumerate(SEGMENTS):
            rgb = (
                ManimColor(BLUE).to_rgb()
                + ManimColor(SEGMENT_COLORS[s]).to_rgb()
                + ManimColor(PURPLE).to_rgb()
            ) / 3
            sums.append(Square(0.5).set_stroke(width=0).set_fill(ManimColor.from_rgb(rgb), 1))
        total = row(sums, -1.75)
        label = Text("= input", font_size=20, color=GREY_B).next_to(total, LEFT, buff=0.25)
        columns = [VGroup(*(d[2][i] for d in drawn)) for i in range(len(TOKENS))]
        self.play(
            *(c.copy().animate.move_to(s).set_opacity(0) for c, s in zip(columns, sums)),
            LaggedStart(*(FadeIn(s) for s in sums), lag_ratio=0.05),
            FadeIn(label),
            run_time=1.5,
        )
        formula = Text(
            "x = token embedding + segment embedding + position embedding",
            font_size=22,
        ).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(formula))
        self.wait(1)
        same = Text(
            "“the” appears twice: same token row, different segment and position",
            font_size=20,
            color=GREY_B,
        ).next_to(formula, UP, buff=0.15)
        self.play(FadeIn(same))
        self.wait(2.5)
        self.play(FadeOut(*self.mobjects))
