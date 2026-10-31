"""Greedy takes the best first word and ends up worse than a beam that kept its options open.

Render: make videos concept=text-generation scene=beam-vs-greedy
"""

from manim import (
    BLUE,
    DOWN,
    GREEN,
    GREY_B,
    ORANGE,
    UP,
    WHITE,
    Create,
    FadeIn,
    FadeOut,
    Indicate,
    Line,
    RoundedRectangle,
    Scene,
    Text,
    VGroup,
)


def node(word: str, x: float, y: float, color: str = GREY_B) -> VGroup:
    box = RoundedRectangle(corner_radius=0.08, width=1.5, height=0.62)
    box.set_stroke(color, 2).set_fill(color, 0.15).move_to([x, y, 0])
    return VGroup(box, Text(word, font_size=26).move_to(box))


def edge(a: VGroup, b: VGroup, p: str) -> VGroup:
    line = Line(a.get_right(), b.get_left(), color=GREY_B, stroke_width=2)
    label = Text(p, font_size=20, color=GREY_B).next_to(line.get_center(), UP, buff=0.05)
    return VGroup(line, label)


class BeamVsGreedy(Scene):
    def construct(self) -> None:
        title = Text("Greedy can walk past the best sentence", font_size=34).to_edge(UP)
        self.play(FadeIn(title))

        start = node("start", -5.0, 0)
        the, a = node("the", -1.6, 1.3), node("a", -1.6, -1.3)
        the_cat, the_dog = node("cat", 2.0, 2.1), node("dog", 2.0, 0.7)
        a_cat, a_dog = node("cat", 2.0, -0.7), node("dog", 2.0, -2.1)
        first = [edge(start, the, "0.5"), edge(start, a, "0.4")]
        second = [
            edge(the, the_cat, "0.4"),
            edge(the, the_dog, "0.35"),
            edge(a, a_cat, "0.9"),
            edge(a, a_dog, "0.1"),
        ]
        self.play(FadeIn(start))
        self.play(*(Create(e) for e in first), FadeIn(the), FadeIn(a))
        self.wait(0.8)

        note = Text("Greedy: take the likeliest word, every time", font_size=24, color=ORANGE)
        note.to_edge(DOWN, buff=0.3)
        self.play(FadeIn(note))
        self.play(the[0].animate.set_stroke(ORANGE, 4).set_fill(ORANGE, 0.3))
        self.play(*(Create(e) for e in second[:2]), FadeIn(the_cat), FadeIn(the_dog))
        self.play(the_cat[0].animate.set_stroke(ORANGE, 4).set_fill(ORANGE, 0.3))
        greedy = Text("0.5 × 0.4 = 0.20", font_size=26, color=ORANGE)
        greedy.next_to(the_cat, buff=0.3)
        self.play(FadeIn(greedy))
        self.wait(1.2)

        beam = Text("Beam of 2: keep both first words alive", font_size=24, color=BLUE)
        beam.to_edge(DOWN, buff=0.3)
        self.play(FadeOut(note), FadeIn(beam))
        self.play(a[0].animate.set_stroke(BLUE, 4).set_fill(BLUE, 0.3))
        self.play(*(Create(e) for e in second[2:]), FadeIn(a_cat), FadeIn(a_dog))
        self.wait(0.6)
        self.play(a_cat[0].animate.set_stroke(GREEN, 4).set_fill(GREEN, 0.35))
        best = Text("0.4 × 0.9 = 0.36", font_size=26, color=GREEN)
        best.next_to(a_cat, buff=0.3)
        self.play(FadeIn(best))
        self.play(Indicate(best, color=GREEN))
        self.wait(1)

        moral = Text(
            "The worse first word led to the better sentence", font_size=28, color=WHITE
        ).to_edge(DOWN, buff=0.3)
        self.play(FadeOut(beam), FadeIn(moral))
        self.wait(3)
