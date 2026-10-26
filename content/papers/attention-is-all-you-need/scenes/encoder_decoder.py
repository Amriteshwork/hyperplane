"""Translation: the encoder reads the whole sentence at once; the decoder writes word by word,
looking back at its own words so far (masked) and at the encoder's vectors (cross-attention).

Render: make videos paper=attention-is-all-you-need scene=encoder-decoder
"""

from manim import (
    BLUE,
    DOWN,
    GREEN,
    GREY_B,
    LEFT,
    UP,
    YELLOW,
    Create,
    FadeIn,
    FadeOut,
    Indicate,
    LaggedStart,
    Line,
    RoundedRectangle,
    Scene,
    Square,
    Text,
    VGroup,
)

ENGLISH = ["the", "cat", "sat", "down"]
GERMAN = ["die", "Katze", "setzte", "sich"]
#: Cross-attention of each German word over the English words (rows sum to 1).
CROSS = [
    [0.85, 0.1, 0.03, 0.02],
    [0.1, 0.85, 0.03, 0.02],
    [0.02, 0.08, 0.8, 0.1],
    [0.02, 0.03, 0.25, 0.7],
]


def word_box(word: str, color: str = GREY_B) -> VGroup:
    label = Text(word, font_size=26)
    box = RoundedRectangle(corner_radius=0.08, width=max(label.width + 0.3, 1.1), height=0.6)
    return VGroup(box.set_stroke(color, 2), label)


class EncoderDecoder(Scene):
    def construct(self) -> None:
        enc_label = Text("Encoder: read the whole sentence", font_size=28, color=BLUE)
        enc_label.to_corner(UP + LEFT)
        english = VGroup(*(word_box(w, BLUE) for w in ENGLISH)).arrange(buff=0.4)
        english.next_to(enc_label, DOWN, buff=0.4).set_x(0)
        self.play(FadeIn(enc_label), LaggedStart(*(FadeIn(w) for w in english), lag_ratio=0.1))

        # Every English word looks at every English word, both ways, in one go.
        vectors = VGroup(
            *(
                Square(0.45).set_stroke(width=0).set_fill(BLUE, 0.8).next_to(w, DOWN, buff=0.7)
                for w in english
            )
        )
        z_label = Text("one vector per word, full of context", font_size=20, color=GREY_B)
        z_label.next_to(vectors, DOWN, buff=0.2)
        self.play(
            *(Indicate(w, color=BLUE) for w in english),
            LaggedStart(*(FadeIn(v, shift=DOWN * 0.3) for v in vectors), lag_ratio=0),
            FadeIn(z_label),
        )
        self.wait(0.5)

        dec_label = Text("Decoder: write the translation, word by word", font_size=28, color=GREEN)
        slots = VGroup(*(word_box(w, GREEN) for w in GERMAN)).arrange(buff=0.4).shift(DOWN * 1.6)
        dec_label.next_to(slots, DOWN, buff=0.3).to_edge(LEFT)
        self.play(FadeIn(dec_label), FadeOut(z_label))

        written: list[VGroup] = []
        for i, slot in enumerate(slots):
            box = slot[0].copy()
            self.play(Create(box), run_time=0.4)
            # Masked self-attention: only the German words already written.
            own = VGroup(
                *(
                    Line(box.get_left(), w.get_right(), path_arc=0.6).set_stroke(GREEN, 2)
                    for w in written
                )
            )
            # Cross-attention: the queries come from here, keys and values from the encoder.
            cross = VGroup(
                *(
                    Line(box.get_top(), v.get_bottom()).set_stroke(YELLOW, 1 + 10 * a, opacity=a)
                    for v, a in zip(vectors, CROSS[i])
                )
            )
            self.play(Create(own, lag_ratio=0), Create(cross, lag_ratio=0), run_time=0.8)
            word = slot[1].copy()
            self.play(FadeIn(word, shift=UP * 0.2), run_time=0.5)
            self.play(FadeOut(own, cross), run_time=0.4)
            written.append(VGroup(box, word))

        rules = VGroup(
            Text("green: looks only at German already written (masked)", font_size=20, color=GREEN),
            Text("yellow: looks back at the English (cross-attention)", font_size=20, color=YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        rules.to_edge(DOWN, buff=0.3).to_edge(LEFT)
        self.play(FadeIn(rules))
        self.wait(2.5)
        self.play(FadeOut(*self.mobjects))
