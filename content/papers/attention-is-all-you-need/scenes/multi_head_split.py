"""Multi-head attention: split a 512-number vector into 8 heads of 64, let each attend its own
way, glue the answers back together and mix them with W_O.

Render: make videos paper=attention-is-all-you-need scene=multi-head-split
"""

from manim import (
    BLUE,
    DOWN,
    GREEN,
    GREY_B,
    LEFT,
    ORANGE,
    PINK,
    PURPLE,
    RED,
    RIGHT,
    TEAL,
    UP,
    YELLOW,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    LaggedStart,
    Line,
    Rectangle,
    Scene,
    Text,
    Transform,
    VGroup,
)

HEADS = 8
CELLS = 16  # each cell stands for 32 of the 512 numbers
COLORS = [BLUE, GREEN, YELLOW, ORANGE, RED, PINK, PURPLE, TEAL]
WORDS = ["the", "animal", "didn't", "cross", "because", "it"]
#: Which word "it" looks at in each head, and what that head seems to be asking.
FOCUS = [
    (1, "what does “it” refer to?"),
    (4, "the word just before"),
    (3, "what's the action?"),
    (5, "itself"),
    (1, ""),
    (0, ""),
    (2, ""),
    (3, ""),
]


def cells(colors: list[str]) -> VGroup:
    return VGroup(
        *(Rectangle(width=0.32, height=0.4).set_stroke(GREY_B, 1).set_fill(c, 0.8) for c in colors)
    ).arrange(buff=0)


class MultiHeadSplit(Scene):
    def construct(self) -> None:
        title = Text("Eight small heads instead of one big one", font_size=36).to_edge(UP)
        vector = cells([c for c in COLORS for _ in range(CELLS // HEADS)])
        vector.next_to(title, DOWN, buff=0.5)
        label = Text("“it”: 512 numbers", font_size=24).next_to(vector, DOWN)
        self.play(FadeIn(title), FadeIn(vector), FadeIn(label))
        top = vector.get_center()
        self.wait(0.5)

        # Split into 8 rows of 64 numbers, one per head.
        rows = VGroup(*(cells([c, c]) for c in COLORS)).arrange(DOWN, buff=0.18)
        rows.to_edge(LEFT, buff=1.2).shift(DOWN * 0.6)
        names = VGroup(
            *(Text(f"head {i + 1}", font_size=18, color=GREY_B).next_to(r, LEFT) for i, r in enumerate(rows))
        )
        pieces = [vector[2 * i : 2 * i + 2] for i in range(HEADS)]
        self.play(
            FadeOut(label),
            *(p.animate.move_to(r) for p, r in zip(pieces, rows)),
            FadeIn(names),
            run_time=1.5,
        )
        note = Text("each head: its own 64 numbers, its own attention", font_size=22)
        note.next_to(title, DOWN, buff=0.3)
        self.play(FadeIn(note))

        # Each head looks at a different word.
        patterns = VGroup()
        for row, color, (target, question) in zip(rows, COLORS, FOCUS):
            dots = VGroup(*(Dot(radius=0.06, color=GREY_B) for _ in WORDS)).arrange(buff=0.35)
            dots.next_to(row, RIGHT, buff=0.6)
            it, seen = dots[-1].copy().set_color(color), dots[target]
            arc = Line(it.get_center(), seen.get_center(), path_arc=-1.2).set_stroke(color, 3)
            ask = Text(question, font_size=18, color=color).next_to(dots, RIGHT, buff=0.4)
            patterns.add(VGroup(dots, it, arc, ask))
        words = VGroup(
            *(Text(w, font_size=14, color=GREY_B).next_to(d, UP, buff=0.1) for w, d in zip(WORDS, patterns[0][0]))
        )
        self.play(FadeIn(words), LaggedStart(*(FadeIn(p[0], p[1]) for p in patterns), lag_ratio=0.05))
        self.play(LaggedStart(*(Create(p[2]) for p in patterns), lag_ratio=0.1))
        self.play(LaggedStart(*(FadeIn(p[3]) for p in patterns), lag_ratio=0.1))
        self.wait(2)

        # Glue the 8 answers back together and mix them.
        glued = cells([c for c in COLORS for _ in range(CELLS // HEADS)]).move_to(top)
        self.play(
            FadeOut(patterns, words, names, note),
            *(p.animate.move_to(glued[2 * i : 2 * i + 2]) for i, p in enumerate(pieces)),
            run_time=1.5,
        )
        mix = Text("× Wᴼ  (mix the heads' answers)", font_size=24).next_to(vector, DOWN, buff=0.4)
        mixed = cells([BLUE] * CELLS).next_to(mix, DOWN, buff=0.4)
        for i, cell in enumerate(mixed):
            cell.set_fill(COLORS[(i * 3) % HEADS], 0.8)
        self.play(FadeIn(mix))
        self.play(Transform(vector.copy(), mixed), run_time=1.2)
        cost = Text("Same cost as one big head:  8 × 64 = 512", font_size=28, color=YELLOW)
        cost.to_edge(DOWN, buff=0.6)
        self.play(FadeIn(cost))
        self.wait(2)
        self.play(FadeOut(*self.mobjects))
