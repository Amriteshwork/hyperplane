"""An RNN needs one step per word, and its memory of early words fades; attention takes one step.

Render: make videos paper=attention-is-all-you-need scene=sequential-vs-parallel
"""

import numpy as np
from manim import (
    BLUE,
    DOWN,
    GREY_B,
    ORIGIN,
    RIGHT,
    UP,
    YELLOW,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    LaggedStart,
    Line,
    RoundedRectangle,
    Scene,
    Text,
    Transform,
    VGroup,
)

WORDS = ["the", "animal", "didn't", "cross", "because", "it", "was", "tired"]
#: How much "it" attends to each word: mostly "animal".
IT_WEIGHTS = [0.03, 0.62, 0.02, 0.03, 0.04, 0.1, 0.06, 0.1]


def word_box(word: str) -> VGroup:
    label = Text(word, font_size=26)
    box = RoundedRectangle(corner_radius=0.08, width=label.width + 0.3, height=0.6)
    box.set_stroke(GREY_B, 2).set_fill(YELLOW, opacity=0)
    return VGroup(box, label)


class SequentialVsParallel(Scene):
    def construct(self) -> None:
        title = Text("An RNN reads one word at a time", font_size=36).to_edge(UP)
        words = VGroup(*(word_box(w) for w in WORDS)).arrange(buff=0.15).shift(UP * 0.6)
        self.play(FadeIn(title), LaggedStart(*(FadeIn(w) for w in words), lag_ratio=0.1))

        memory = RoundedRectangle(corner_radius=0.1, width=1.0, height=0.5)
        memory.set_stroke(YELLOW).set_fill(YELLOW, opacity=0.25)
        memory.next_to(words[0], DOWN, buff=0.8)
        memory_label = Text("memory", font_size=20, color=YELLOW).next_to(memory, DOWN)
        memory_label.add_updater(lambda m: m.next_to(memory, DOWN))
        # What the memory still holds about "animal": it fades with every later word.
        animal = Dot(color=BLUE, radius=0.12).set_opacity(0)
        animal.add_updater(lambda m: m.move_to(memory))
        steps = Text("step 1", font_size=30).to_edge(DOWN)
        self.play(FadeIn(memory, memory_label, steps))

        for i, word in enumerate(words):
            opacity = 1.0 if i == 1 else animal.get_fill_opacity() * 0.6
            self.play(
                memory.animate.next_to(word, DOWN, buff=0.8),
                word[0].animate.set_fill(YELLOW, opacity=0.35),
                animal.animate.set_opacity(opacity),
                Transform(steps, Text(f"step {i + 1}", font_size=30).to_edge(DOWN)),
                run_time=0.6,
            )
        note = Text(
            "8 steps, one after another. By “it”, “animal” has almost faded.",
            font_size=26,
        ).next_to(steps, UP)
        self.play(FadeIn(note))
        self.wait(1.5)

        # Attention: put the words on a circle and connect every pair at once.
        self.play(
            FadeOut(memory, memory_label, animal, note, steps),
            Transform(
                title, Text("Attention: every word looks at every word", font_size=36).to_edge(UP)
            ),
            *(w[0].animate.set_fill(opacity=0) for w in words),
        )
        angles = np.linspace(0, 2 * np.pi, len(WORDS), endpoint=False) + np.pi / 2
        self.play(
            *(
                w.animate.move_to(ORIGIN + 2.0 * np.array([np.cos(a), np.sin(a), 0]) + DOWN * 0.3)
                for w, a in zip(words, angles)
            )
        )
        chords = VGroup(
            *(
                Line(words[i].get_center(), words[j].get_center(), stroke_width=1.5)
                .set_stroke(GREY_B, opacity=0.5)
                for i in range(len(WORDS))
                for j in range(i + 1, len(WORDS))
            )
        )
        chords.set_z_index(-1)
        one_step = Text("1 step", font_size=30, color=YELLOW).to_corner(UP + RIGHT).shift(DOWN * 0.8)
        self.play(Create(chords, lag_ratio=0), FadeIn(one_step), run_time=1.2)
        self.wait(0.5)

        # Where "it" looks: line thickness is its attention weight.
        it = words[WORDS.index("it")]
        focus = VGroup(
            *(
                Line(it.get_center(), w.get_center(), stroke_width=2 + 24 * weight, color=YELLOW)
                .set_z_index(-1)
                for w, weight in zip(words, IT_WEIGHTS)
                if w is not it
            )
        )
        self.play(
            chords.animate.set_stroke(opacity=0.12),
            it[0].animate.set_fill(YELLOW, opacity=0.35),
            Create(focus, lag_ratio=0),
        )
        caption = Text("“it” reaches “animal” directly, in one step", font_size=26)
        caption.to_edge(DOWN)
        self.play(FadeIn(caption))
        self.wait(2)
