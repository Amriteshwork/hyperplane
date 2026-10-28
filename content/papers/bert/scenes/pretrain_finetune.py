"""Pretrain once, fine-tune many times: one expensive model on billions of words, then a
cheap copy per task, each with a tiny head on the outputs it needs.

Render: make videos paper=bert scene=pretrain-finetune
"""

from manim import (
    BLUE,
    DOWN,
    GREEN,
    GREY_B,
    LEFT,
    ORANGE,
    PURPLE,
    RIGHT,
    UP,
    YELLOW,
    Arrow,
    FadeIn,
    FadeOut,
    LaggedStart,
    RoundedRectangle,
    Scene,
    Square,
    Text,
    VGroup,
)

TASKS = [
    ("Is this review positive?", "reads [CLS] only", "positive / negative", YELLOW, [0]),
    (
        "Find the names and places",
        "reads every token",
        "PERSON · PLACE · other",
        GREEN,
        [1, 2, 3, 4],
    ),
    ("Where is the answer?", "reads every token", "start ▸ … ◂ end", ORANGE, [2, 4]),
]


def bert_box(width: float = 2.4, height: float = 1.0, size: int = 26) -> VGroup:
    box = RoundedRectangle(corner_radius=0.15, width=width, height=height).set_stroke(BLUE, 3)
    box.set_fill(BLUE, 0.12)
    return VGroup(box, Text("BERT", font_size=size, color=BLUE).move_to(box))


def outputs(model: VGroup, used: list[int], color: str) -> VGroup:
    """Five output vectors on top of a model, the ones a task reads in its colour."""
    squares = VGroup(
        *(
            Square(0.22)
            .set_stroke(width=0)
            .set_fill(color if i in used else GREY_B, 1 if i in used else 0.3)
            for i in range(5)
        )
    ).arrange(buff=0.1)
    return squares.next_to(model, UP, buff=0.1)


class PretrainFinetune(Scene):
    def construct(self) -> None:
        title = Text("Step 1: pretrain once", font_size=34).to_edge(UP)
        books = Text(
            "3.3 billion words\nWikipedia + books\nno labels", font_size=24, line_spacing=0.8
        )
        books.to_edge(LEFT, buff=0.8).shift(UP * 0.3)
        model = bert_box(3.0, 1.4, 34).shift(RIGHT * 2.6 + UP * 0.3)
        arrow = Arrow(books.get_right(), model.get_left(), buff=0.3, color=GREY_B)
        tasks = Text(
            "fill in the blanks\n+ next sentence", font_size=20, color=GREY_B, line_spacing=0.8
        )
        tasks.next_to(arrow, DOWN, 0.15)
        self.play(FadeIn(title), FadeIn(books))
        self.play(FadeIn(arrow), FadeIn(tasks), FadeIn(model))
        cost = Text("4 days on 16 TPU chips: expensive, but done once", font_size=22, color=PURPLE)
        cost.next_to(model, DOWN, buff=0.9)
        self.play(FadeIn(cost))
        self.wait(1.5)

        # Step 2: copy the same weights into every task and add a small head.
        new_title = Text("Step 2: fine-tune a copy for each task", font_size=34).to_edge(UP)
        self.play(FadeOut(title, books, arrow, tasks, cost), FadeIn(new_title))
        copies = VGroup(*(bert_box() for _ in TASKS)).arrange(buff=1.2).shift(DOWN * 0.9)
        self.play(*(model.copy().animate.become(c) for c in copies), FadeOut(model), run_time=1.2)
        self.add(copies)

        for (question, reads, head, color, used), copy in zip(TASKS, copies):
            ask = Text(question, font_size=20).next_to(copy, DOWN, buff=0.3)
            outs = outputs(copy, used, color)
            how = Text(reads, font_size=16, color=color).next_to(outs, UP, buff=0.12)
            head_box = RoundedRectangle(corner_radius=0.08, width=2.4, height=0.5).set_stroke(
                color, 2
            )
            head_box.next_to(how, UP, buff=0.15)
            head_label = Text(head, font_size=16, color=color).move_to(head_box)
            head_label.scale_to_fit_width(min(head_label.width, head_box.width - 0.2))
            self.play(
                FadeIn(ask),
                LaggedStart(
                    FadeIn(outs), FadeIn(how), FadeIn(head_box), FadeIn(head_label), lag_ratio=0.3
                ),
            )
            self.wait(0.5)

        small = Text(
            "each head: a few thousand new weights · about an hour of training",
            font_size=22,
            color=GREY_B,
        ).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(small))
        self.wait(2.5)
        self.play(FadeOut(*self.mobjects))
