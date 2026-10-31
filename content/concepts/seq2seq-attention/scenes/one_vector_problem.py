"""A long sentence squeezed through one vector, then the decoder looking back at every word instead.

Render: make videos concept=seq2seq-attention scene=one-vector-problem
"""

from manim import (
    BLUE,
    DOWN,
    GREEN,
    GREY_B,
    ORANGE,
    UP,
    WHITE,
    Arrow,
    FadeIn,
    FadeOut,
    GrowArrow,
    LaggedStart,
    RoundedRectangle,
    Scene,
    Square,
    Text,
    Transform,
    VGroup,
)

ENGLISH = ["figs", "jam", "rice", "kale", "eggs", "soup"]
GERMAN = ["feigen", "marmelade", "reis", "grünkohl", "eier", "suppe"]
#: Cross-attention weights the trained model in this page's code produced.
WEIGHTS = [
    [0.87, 0.09, 0.01, 0.01, 0.01, 0.01],
    [0.16, 0.71, 0.11, 0.01, 0.01, 0.00],
    [0.00, 0.22, 0.63, 0.13, 0.01, 0.01],
    [0.00, 0.00, 0.29, 0.58, 0.10, 0.02],
    [0.00, 0.00, 0.00, 0.25, 0.56, 0.18],
    [0.00, 0.00, 0.00, 0.00, 0.17, 0.83],
]
STEP = 1.75


def word(text: str, x: float, y: float, color: str) -> VGroup:
    box = RoundedRectangle(corner_radius=0.08, width=1.6, height=0.6)
    box.set_stroke(color, 2).set_fill(color, 0.12).move_to([x, y, 0])
    label = Text(text, font_size=22)
    label.scale_to_fit_width(min(label.width, 1.4))
    return VGroup(box, label.move_to(box))


class OneVectorProblem(Scene):
    def construct(self) -> None:
        title = Text("Translating through one vector", font_size=34).to_edge(UP)
        self.play(FadeIn(title))

        xs = [(i - 2.5) * STEP for i in range(6)]
        english = VGroup(*(word(w, x, 1.9, BLUE) for w, x in zip(ENGLISH, xs)))
        self.play(LaggedStart(*(FadeIn(w, shift=DOWN * 0.2) for w in english), lag_ratio=0.15))

        vector = Square(side_length=0.75).set_stroke(ORANGE, 3).set_fill(ORANGE, 0.5)
        vector.move_to([0, 0.2, 0])
        label = Text("one vector", font_size=22, color=ORANGE).next_to(vector, buff=0.25)
        arrows = [Arrow(w.get_bottom(), vector.get_top(), buff=0.08, color=GREY_B) for w in english]
        self.play(*(GrowArrow(a) for a in arrows), FadeIn(vector), FadeIn(label))
        german = VGroup(*(word(w, x, -1.7, GREEN) for w, x in zip(GERMAN, xs)))
        out = [Arrow(vector.get_bottom(), w.get_top(), buff=0.08, color=GREY_B) for w in german]
        self.play(*(GrowArrow(a) for a in out), FadeIn(german))
        problem = Text("6 words or 60: it's the same size", font_size=26, color=ORANGE)
        problem.to_edge(DOWN, buff=0.35)
        self.play(FadeIn(problem))
        self.wait(2.5)

        title2 = Text("Attention: keep every state, and look back", font_size=34).to_edge(UP)
        self.play(
            Transform(title, title2),
            *(FadeOut(a) for a in [*arrows, *out]),
            FadeOut(vector),
            FadeOut(label),
            FadeOut(problem),
        )
        for t, row in enumerate(WEIGHTS):
            links = [
                Arrow(
                    english[s].get_bottom(),
                    german[t].get_top(),
                    buff=0.08,
                    stroke_width=max(0.5, 12 * w),
                    color=WHITE,
                    stroke_opacity=min(1.0, 0.1 + w),
                    max_tip_length_to_length_ratio=0.05,
                )
                for s, w in enumerate(row)
                if w >= 0.05
            ]
            highlight = german[t][0].copy().set_stroke(GREEN, 5)
            self.play(*(GrowArrow(a) for a in links), FadeIn(highlight), run_time=0.7)
            self.wait(0.5)
            self.play(*(FadeOut(a) for a in links), FadeOut(highlight), run_time=0.3)

        moral = Text(
            "These are the weights a trained model really used. Nobody taught it the alignment.",
            font_size=22,
        ).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(moral))
        self.wait(3)
