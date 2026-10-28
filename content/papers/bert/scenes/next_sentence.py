"""Next sentence prediction: half the pairs are real neighbours, half are random; BERT reads
the pair and answers IsNext or NotNext from the [CLS] vector.

Render: make videos paper=bert scene=next-sentence
"""

from manim import (
    BLUE,
    DOWN,
    GREEN,
    GREY_B,
    RED,
    RIGHT,
    UP,
    YELLOW,
    Arrow,
    FadeIn,
    FadeOut,
    Indicate,
    RoundedRectangle,
    Scene,
    Square,
    Text,
    VGroup,
)

PAIRS = [
    ("The man went to the store.", "He bought a gallon of milk.", "IsNext", GREEN),
    ("The man went to the store.", "Penguins are flightless birds.", "NotNext", RED),
]


def sentence_card(text: str, tag: str, color: str) -> VGroup:
    label = Text(text, font_size=22)
    box = RoundedRectangle(corner_radius=0.1, width=label.width + 0.5, height=0.7)
    tag_text = Text(tag, font_size=18, color=color).next_to(
        box, UP, buff=0.05, aligned_edge=[-1, 0, 0]
    )
    return VGroup(box.set_stroke(color, 2), label, tag_text)


class NextSentence(Scene):
    def construct(self) -> None:
        title = Text("Task 2: does sentence B follow sentence A?", font_size=32).to_edge(UP)
        self.play(FadeIn(title))
        coin = Text(
            "50%: B is the real next sentence    50%: B is a random one", font_size=22, color=GREY_B
        )
        coin.next_to(title, DOWN, buff=0.25)
        self.play(FadeIn(coin))

        model = RoundedRectangle(corner_radius=0.2, width=2.6, height=1.6).set_stroke(BLUE, 3)
        model_label = Text("BERT", font_size=32, color=BLUE).move_to(model)
        bert = VGroup(model, model_label).shift(RIGHT * 2.2 + DOWN * 0.6)
        self.play(FadeIn(bert))

        for a, b, answer, color in PAIRS:
            cards = VGroup(
                sentence_card(a, "sentence A", GREY_B), sentence_card(b, "sentence B", YELLOW)
            )
            cards.arrange(DOWN, buff=0.5).to_edge([-1, 0, 0], buff=0.5).shift(DOWN * 0.6)
            self.play(FadeIn(cards, shift=RIGHT * 0.3))
            into = Arrow(cards.get_right(), bert.get_left(), buff=0.15, color=GREY_B)
            packed = Text("[CLS] A [SEP] B [SEP]", font_size=18, color=GREY_B).next_to(
                into, UP, buff=0.05
            )
            self.play(FadeIn(into), FadeIn(packed))
            self.play(Indicate(model, color=BLUE))

            cls = Square(0.45).set_stroke(width=0).set_fill(YELLOW, 1).next_to(bert, DOWN, buff=0.5)
            cls_label = Text("[CLS] vector", font_size=18, color=YELLOW).next_to(
                cls, RIGHT, buff=0.15
            )
            verdict = Text(answer, font_size=30, color=color).next_to(cls, DOWN, buff=0.35)
            self.play(FadeIn(cls, shift=DOWN * 0.2), FadeIn(cls_label))
            self.play(FadeIn(verdict, scale=1.2))
            self.wait(1.5)
            self.play(FadeOut(cards, into, packed, cls, cls_label, verdict))

        note = (
            VGroup(
                Text("Free labels: any book or article makes millions of pairs", font_size=22),
                Text(
                    "Later, RoBERTa found BERT did just as well without this task",
                    font_size=20,
                    color=GREY_B,
                ),
            )
            .arrange(DOWN, buff=0.15)
            .to_edge(DOWN, buff=0.4)
        )
        self.play(FadeIn(note))
        self.wait(2.5)
        self.play(FadeOut(*self.mobjects))
