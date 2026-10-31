"""A window slides along one sentence, making training pairs. Shared neighbours pull two words together.

Render: make videos concept=word-embeddings scene=sliding-window
"""

from manim import (
    BLUE,
    DOWN,
    GREEN,
    GREY_B,
    ORANGE,
    UP,
    Arrow,
    Dot,
    FadeIn,
    FadeOut,
    GrowArrow,
    RoundedRectangle,
    Scene,
    Text,
    Transform,
    VGroup,
)

WORDS = ["the", "cat", "hunts", "at", "night"]
STEP = 1.9


def word(text: str, x: float, y: float) -> VGroup:
    box = RoundedRectangle(corner_radius=0.08, width=1.6, height=0.65).move_to([x, y, 0])
    box.set_stroke(GREY_B, 2)
    return VGroup(box, Text(text, font_size=26).move_to(box))


class SlidingWindow(Scene):
    def construct(self) -> None:
        title = Text("Slide a window, get a dataset", font_size=34).to_edge(UP)
        self.play(FadeIn(title))
        xs = [(i - 2) * STEP for i in range(len(WORDS))]
        cells = VGroup(*(word(w, x, 1.8) for w, x in zip(WORDS, xs)))
        self.play(FadeIn(cells))

        pairs = VGroup()
        for center in range(len(WORDS)):
            around = [j for j in range(center - 2, center + 3) if 0 <= j < len(WORDS) and j != center]
            marks = [cells[center][0].copy().set_stroke(ORANGE, 5)]
            marks += [cells[j][0].copy().set_stroke(BLUE, 4) for j in around]
            line = Text(
                "  ".join(f"({WORDS[center]}, {WORDS[j]})" for j in around), font_size=19, color=GREY_B
            )
            line.move_to([0, 0.75 - center * 0.42, 0])
            pairs.add(line)
            self.play(*(FadeIn(m) for m in marks), FadeIn(line), run_time=0.5)
            self.wait(0.35)
            self.play(*(FadeOut(m) for m in marks), run_time=0.2)
        note = Text("One short sentence: 14 training examples, and no labels", font_size=24, color=ORANGE)
        note.to_edge(DOWN, buff=0.3)
        self.play(FadeIn(note))
        self.wait(2)

        title2 = Text("Shared neighbours pull words together", font_size=34).to_edge(UP)
        self.play(Transform(title, title2), FadeOut(cells), FadeOut(pairs), FadeOut(note))
        spots = {"sleeps": (-1, 2.0), "runs": (1, 2.0), "hunts": (0, -2.2)}
        neighbours = VGroup()
        for name, (x, y) in spots.items():
            neighbours.add(VGroup(Dot([x, y, 0], color=BLUE), Text(name, font_size=22, color=BLUE).next_to([x, y, 0], UP, buff=0.1)))
        cat_dot = Dot([-5, -0.2, 0], color=GREEN, radius=0.12)
        dog_dot = Dot([5, 0.2, 0], color=ORANGE, radius=0.12)
        cat = Text("cat", font_size=26, color=GREEN).next_to(cat_dot, DOWN, buff=0.1)
        dog = Text("dog", font_size=26, color=ORANGE).next_to(dog_dot, DOWN, buff=0.1)
        self.play(FadeIn(neighbours), FadeIn(cat_dot), FadeIn(dog_dot), FadeIn(cat), FadeIn(dog))
        arrows = [
            Arrow(d.get_center(), [x, y, 0], color=GREY_B, buff=0.25, stroke_width=2)
            for d in (cat_dot, dog_dot)
            for x, y in spots.values()
        ]
        self.play(*(GrowArrow(a) for a in arrows))
        self.wait(1)
        self.play(*(FadeOut(a) for a in arrows))
        self.play(
            cat_dot.animate.move_to([-0.35, -0.1, 0]),
            cat.animate.move_to([-0.35, -0.55, 0]),
            dog_dot.animate.move_to([0.35, 0.1, 0]),
            dog.animate.move_to([0.35, -0.3, 0]),
            run_time=2.5,
        )
        moral = Text(
            "They may never appear together. They just keep the same company.", font_size=26
        ).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(moral))
        self.wait(3)
