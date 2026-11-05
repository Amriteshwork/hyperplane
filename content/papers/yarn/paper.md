---
title: YaRN, explained
summary: The 2023 paper that stretched Llama 2 from 4,000 to 128,000 tokens with 400 steps of fine-tuning, by slowing down RoPE's clocks one at a time. Every formula derived from scratch, with the clocks animated and a figure to compare the methods side by side.
order: 4
source: peng-yarn
concepts: [vectors, softmax, self-attention, positional-encoding, transformer, language-modeling, text-generation]
track: transformers
videos:
  seen-angles:
    title: "Interpolate, don't extrapolate: keep every clock inside the angles it knows"
    youtube: TGrCxOkss_Y
  clocks-by-parts:
    title: "NTK-by-parts: leave the fast clocks alone, slow down the slow ones"
    youtube: P_pBg0SqqSA
  sharper-softmax:
    title: "More tokens, blurrier attention: why YaRN turns up the sharpness"
    youtube: bvGvbY1AHTc
---

Every RoPE model has a **wall**. Llama 2 was trained on texts of 4,096 tokens. Hand it a
10,000-token document and it doesn't get a bit worse at the end: it **falls apart**, producing
nonsense once it passes the length it was trained on. Retraining a 7-billion-parameter model on
long texts from scratch costs a fortune. So in 2023 a question took over the open-source
community: **can you teach an already-trained model to read further, cheaply?**

The best answer came out of that community itself. Its ideas first appeared as Reddit posts, blogs
and GitHub code by people known online as *bloc97*, *emozilla* and *kaiokendev*. In August 2023, Bowen
Peng, Jeffrey Quesnelle, Honglu Fan and Enrico Shippole (Nous Research, EleutherAI and the
University of Geneva) wrote it up properly, added one more trick, and called the result **YaRN**:
*Yet another RoPE extensioN method*. With it, they took Llama 2 from 4,096 tokens to **128,000**,
fine-tuning on less than a thousandth of the original training data. YaRN was published at ICLR
2024, and models such as Qwen 2.5 and DeepSeek-V3 use it to reach their long contexts.

The whole paper is about one thing: **RoPE's clocks**. If you haven't read it yet, the
[RoPE walkthrough](/papers/rope) builds them from scratch. Here's the one-line version: RoPE splits
every query and key into pairs of numbers, and turns each pair like a clock hand, by an angle
proportional to the word's position. The first clock is fast (1 radian per word); each next one is
slower, down to about 1/10,000 radian per word. YaRN's question is: **to read $s$ times further,
how should we slow each clock down?**

**How to read this page.** Sections follow the paper and keep its numbering. Each has **the paper
says** (a short quote), **in plain words**, the **maths derived step by step**, and **why it
mattered**. One figure lets you compare all the methods on Llama 2's actual 64 clocks; three
animations show the key ideas moving.

## Abstract and 1 · The wall at the end of the context

> **The paper says:** *"Rotary Position Embeddings (RoPE) have been shown to effectively encode
> positional information in transformer-based language models. However, these models fail to
> generalize past the sequence length they were trained on."*

**In plain words.** The **context window** is how many tokens a model can look at once. It's set
by training: Llama 2 saw texts of up to $L = 4{,}096$ tokens, so 4,096 is its window. Push past it
and quality collapses.

That should be surprising. RoPE is a **relative** position scheme: attention scores only depend on
how far apart two words are, never on where they are (the RoPE page proves it). Distance 10 at
position 5,000 looks exactly like distance 10 at position 50. So why would position 5,000 be a
problem?

Because of **long distances**, not late positions. At position 10,000, a word can look back
10,000 tokens, and the model has never seen a gap longer than 4,096. On the fast clocks that
doesn't matter: they've turned hundreds of times during training and have seen every angle. But the
slowest clocks never even completed one turn in 4,096 tokens. For them, a gap of 10,000 means an
**angle they have never seen**, and the model has no idea what to do with it. The paper puts it
bluntly:

> *"…in practice, RoPE does not only encode relative position."*

**Why it mattered.** Once you see the problem as "some clocks are asked for angles they never
saw", the fixes become a question you can reason about precisely, clock by clock. That is the whole
paper.

**Go deeper:** [Positional encoding](/concepts/positional-encoding) ·
[RoPE, explained](/papers/rope)

## 0 · A refresher: clocks, wavelengths and turns

This section isn't in the paper; it collects the three numbers the paper reasons with.

**Speed.** Clock $d$ (counting from 0) turns $\theta_d$ radians per token:

$$\theta_d = b^{-2d/|D|},$$

where $b = 10{,}000$ is the **base** and $|D|$ is the number of numbers per head (128 in Llama 2,
so 64 clocks). Clock 0 turns at $b^0 = 1$ radian per token; the last turns at about $1/b$.

**Wavelength.** How many tokens a clock needs for one full turn ($2\pi$ radians):

$$\lambda_d = \frac{2\pi}{\theta_d} = 2\pi\, b^{2d/|D|}.$$

**Which clocks are in danger?** Here's the question to ask. A longer text means bigger gaps, and a
bigger gap turns a clock further. When does that produce an angle the model has never seen?

Angles live on a circle, so turning by $2\pi$ lands you back where you started. If a clock went
**all the way round** at least once during training, then every angle on its face has already
come up, and no gap, however long, can show it anything new. If it **didn't**, it has only ever
seen the slice from 0 up to $L\,\theta_d$, and a long enough gap pushes it off that slice.

So the one number that matters is how many full turns a clock made across the training length
$L$:

$$r(d) = \frac{L}{\lambda_d} = \frac{L\,\theta_d}{2\pi}.$$

$r \ge 1$: safe, every angle is familiar. $r < 1$: in danger. Here it is for Llama 2
($L = 4{,}096$):

| Clock | Wavelength $\lambda$ (tokens) | Turns in training $r$ | Has it seen every angle? |
|---|---|---|---|
| 0 (fastest) | 6.3 | 652 | yes, hundreds of times |
| 20 | 112 | 37 | yes, dozens of times |
| 32 | 628 | 6.5 | yes, a few times |
| 46 | 4,712 | 0.87 | **no**: never a full turn |
| 63 (slowest) | 54,410 | 0.075 | **no**: only 1/13 of a turn |

The fast clocks are safe hundreds of times over. The danger is all at the slow end: the last 18
clocks never finished a turn, and the slowest saw only a thin wedge of its face.

**The stretch.** To read $L'$ tokens instead of $L$, the **scale factor** is

$$s = \frac{L'}{L}.$$

Going from 4,096 to 65,536 tokens is $s = 16$; to 131,072 tokens, $s = 32$.

## 2.1 and 2.2 · Two knobs: move the positions, or change the speeds

> **The paper says:** *"…a modification $f'_{W}$ can take the following form:
> $f'_{W}(x_m, m, \theta) = f_W(x_m, g(m), h(\theta))$."*

**In plain words.** RoPE turns the query or key of the word at position $m$ by angle $m\,\theta_d$
on clock $d$. Every context-extension method in this paper changes that angle in one of two ways:

- **$g(m)$: relabel the positions.** Pretend word $m$ sits somewhere else, for example at $m/s$.
- **$h(\theta)$: change the clock speeds.** Make some clocks turn more slowly.

The new angle is $g(m) \cdot h(\theta_d)$. Notice something: dividing every position by $s$ gives
the angle $(m/s)\,\theta_d = m\,(\theta_d/s)$, which is the same as **slowing every clock by $s$**.
The two knobs are really one. So from here on, think of every method as an answer to a single
question:

> **How much slower should each clock turn?**

Write the answer as a **slowdown factor** for each clock: $\theta_d / h(\theta_d)$. "1×" means the
clock is untouched; "16×" means it turns 16 times more slowly.

**Why it mattered.** Putting all methods in one frame turns a zoo of tricks into a comparison you
can draw: one bar per clock, its height the slowdown. The figure in section 3.2 does exactly that.

## 2.3 · Position Interpolation: squeeze everything

> **Chen et al. (2023) say:** position interpolation avoids *"extrapolating beyond the trained
> context length which may lead to catastrophically high attention scores that completely ruin the
> self-attention mechanism."*

**In plain words.** The first fix, **Position Interpolation (PI)**, came from Meta (Chen et al.)
and, at the same time, from the hobbyist *kaiokendev*. Don't let positions go beyond what the model
knows: **squeeze** them. To fit $L' = sL$ tokens into a model trained on $L$, divide every position
by $s$:

$$g(m) = \frac{m}{s}, \qquad h(\theta) = \theta .$$

Position 65,536 becomes "position 4,096". Position 3 becomes "position 0.1875". Every clock is
slowed by the same factor $s$.

**Why it can't leave the trained range.** The largest angle any clock reaches at the new length
$L'$ is

$$\frac{L'}{s}\,\theta_d = \frac{sL}{s}\,\theta_d = L\,\theta_d ,$$

exactly the largest angle it reached in training. No clock is ever asked for an angle it hasn't
seen. The animation below shows the two options side by side on one slow clock: **extrapolation**
runs off into unseen territory; **interpolation** stays inside.

::video[seen-angles]

PI works: with about 1,000 steps of fine-tuning, LLaMA read 32,768 tokens. But it has a price,
and the next section is about it.

**Why it mattered.** "Interpolate, don't extrapolate" is the principle every later method keeps.
The question left is whether *every* clock needs the same treatment.

## 3.1 · Loss of high frequency information: "NTK-aware" interpolation

> **The paper says:** *"…it removes the high frequency components of RoPE."*

**In plain words.** Look at what PI does to the **fastest** clock. In training it turned 1 radian
per token: neighbouring words sat a full radian apart, and the model used that to tell "the word
right before me" from "two words back". After PI with $s = 16$, it turns $1/16 \approx 0.06$
radians per token. Neighbours are now almost on top of each other. The model's **fine sense of
nearby order goes blurry**, like shrinking a photo and losing its small print.

The paper links this to a result about neural networks (Tancik et al., 2020, using **Neural
Tangent Kernel** theory, hence "NTK"): networks struggle to learn fine detail when their inputs
lack **high-frequency** components. RoPE's fast clocks *are* those high frequencies, and PI throws
them away. In practice, PI fine-tunes stopped working beyond about $s = 8$:

> *"…only able to achieve a scaling factor of roughly $s=8$ before the LLM's outputs starts to
> degrade, even after fine-tuning."*

**Let's design the fix ourselves.** PI slows every clock by $s$. What do we actually need?

- **The slowest clock must be slowed by $s$.** It's the one most in danger (section 0), and $s$ is
  exactly what PI uses to keep it on its slice.
- **The fastest clock should be slowed by 1×**, not at all. It's the one PI damaged, and it has
  turned hundreds of times, so it doesn't need protecting.

What about the 62 clocks in between? We need some slowdown that goes from $1$ at clock 0 to $s$ at
the last clock, $d_{\max} = |D|/2 - 1$. A straight line from 1 to $s$ would be the obvious first
guess, but look at how the clocks are spaced. Each one is slower than the one before by the same
**ratio**:

$$\frac{\theta_{d+1}}{\theta_d} = \frac{b^{-2(d+1)/|D|}}{b^{-2d/|D|}} = b^{-2/|D|}.$$

The clocks are evenly spaced on a **log scale**, like the keys of a piano. The slowdown that fits
that spacing is also even on a log scale: every step multiplies it by the same amount. Going from
1 to $s$ in $d_{\max}$ equal multiplications gives

$$\text{slowdown}(d) = s^{\,d/d_{\max}} = s^{\frac{2d}{|D|-2}} .$$

**Now the surprise.** Write down clock $d$'s new speed and collect the powers:

$$\theta_d \cdot s^{-\frac{2d}{|D|-2}}
= b^{-\frac{2d}{|D|}} \cdot \Big(s^{\frac{|D|}{|D|-2}}\Big)^{-\frac{2d}{|D|}}
= \Big(b \cdot s^{\frac{|D|}{|D|-2}}\Big)^{-\frac{2d}{|D|}} .$$

That's exactly RoPE's own formula, $b'^{\,-2d/|D|}$, with a **different base**:

$$\boxed{\;b' = b \cdot s^{\frac{|D|}{|D|-2}}\;}$$

We set out to design a new slowdown for every clock and ended up changing **one number in a
config file**. That's why the idea spread so fast: it was first posted on Reddit as a one-line
change anyone could try. For Llama 2 ($|D| = 128$, $b = 10{,}000$) and $s = 16$:
$b' = 10{,}000 \times 16^{1.016} \approx 167{,}000$. That's all "NTK-aware scaling" is: a bigger
base. **Code Llama** used the same idea with a base of 1,000,000, and Meta's long-context Llama 2
paper called it "adjusted base frequency".

**The catch: some clocks now extrapolate.** Take a clock in the slow half that never completed a
turn in training ($r < 1$). At the new length $sL$, its largest angle is

$$sL \cdot \theta_d\, s^{-\frac{2d}{|D|-2}} = L\theta_d \cdot s^{\,1 - \frac{2d}{|D|-2}} .$$

The exponent $1 - \frac{2d}{|D|-2}$ is **positive for every clock except the very last one**. So
each of those slow clocks reaches **beyond** $L\theta_d$, the largest angle it saw in training.
For Llama 2 with $s = 16$, that's clocks 46 to 62: seventeen clocks pushed up to **2.1 times** past
anything they've seen. That's extrapolation sneaking back in, and it's why NTK-aware models work
well *without* fine-tuning but fine-tune worse than PI. It also makes the right base hard to
choose:

> *"The best base to use for "NTK-aware" interpolation usually has to be found empirically…"*

**Why it mattered.** NTK-aware scaling showed that **different clocks deserve different
treatment**. It just used a blunt tool, one base for all, to do it.

## 3.2 · Loss of relative local distances: "NTK-by-parts" interpolation

> **The paper says:** *"…it is important to not touch the dimensions that only encode relative
> positional information, as they are crucial for the network to distinguish the relative order of
> nearby tokens."*

**In plain words.** NTK-aware failed because it chose each clock's slowdown by its **place in the
line-up** (clock 0, clock 1, …). But section 0 already told us what actually matters: **has the
clock seen every angle?** So let's start again, and decide each clock's speed from its turns in
training, $r$.

**A clock that never finished a turn ($r < 1$).** It has seen angles from 0 up to $L\,\theta_d$.
Give it a new speed $h$. At the new length $sL$ it reaches $sL \cdot h$, and we need that to stay
on the familiar slice:

$$sL \cdot h \le L\,\theta_d \quad\Longleftrightarrow\quad h \le \frac{\theta_d}{s} .$$

So it must be slowed by **at least** $s$. Slowing it more would only squeeze neighbouring words
together for nothing, so the best choice is **exactly** $s$: $h = \theta_d/s$, PI's speed.

**A clock that went all the way round ($r \ge 1$).** No angle can surprise it, so there's no
constraint at all. Slowing it can only blur nearby words, as PI showed. Best choice: **leave it
alone**, $h = \theta_d$.

That gives a sharp rule: slow by $s$ if $r < 1$, otherwise don't touch it. Now test it on the
clocks right at the edge. A clock with $r = 0.99$ gets slowed 16×; its neighbour with $r = 1.01$
isn't touched at all. Two almost identical clocks treated in opposite ways. And is a clock that
made **one or two** turns really "safe"? It has seen each angle only once or twice, at specific
positions in the text, so the model may still be reading where it is from those angles, not just
how far apart two words are. The paper makes this point about clocks that never finish a turn:

> *"…every other token during pre-training has an unique distance to it, which the neural network
> can use to determine its absolute position."*

For a clock with only a few turns, that's nearly as true. Only clocks with **many** turns are
purely about relative distance, the ones the paper says not to touch.

**So soften the edge.** Instead of one cut-off, use two: below $\alpha$ turns, slow fully; above
$\beta$ turns, leave alone; in between, mix the two speeds. The simplest mix is a straight line, a
**ramp** $\gamma$ that goes from 0 ("slow fully") to 1 ("leave alone") as $r$ goes from $\alpha$ to
$\beta$:

$$\gamma(r) = \begin{cases} 0 & r < \alpha \\[2pt] \dfrac{r - \alpha}{\beta - \alpha} & \alpha \le r \le \beta \\[6pt] 1 & r > \beta \end{cases}$$

and mix the two speeds with it:

$$h(\theta_d) = \big(1 - \gamma(r(d))\big)\,\frac{\theta_d}{s} + \gamma(r(d))\,\theta_d .$$

When $\gamma = 0$ this is $\theta_d/s$ (PI's speed); when $\gamma = 1$ it's $\theta_d$ (untouched).

**Where do the edges go?** Our argument fixes one of them: $\alpha = 1$, since anything under one
turn **must** be slowed by $s$. The other one can't be derived; "how many turns count as
plenty?" is a question about how a trained model behaves, so the authors tried values and
measured:

> *"The values of $\alpha$ and $\beta$ should be tuned on a case-by-case basis. For example, we
> have found experimentally that for the Llama family of models, good values for $\alpha$ and
> $\beta$ are $\alpha=1$ and $\beta=32$."*

**Worked example: Llama 2.** With $L = 4{,}096$:

- clocks 0–20 make more than 32 turns: **21 clocks untouched**;
- clocks 46–63 make less than one turn: **18 clocks slowed by $s$**;
- the 25 clocks in between get a blend. A clock with $r = 16.5$ sits halfway up the ramp,
  $\gamma = 0.5$, so at $s = 16$ its speed becomes $\theta\,(0.5/16 + 0.5) \approx 0.53\,\theta$:
  about 1.9× slower.

And no clock extrapolates. The slow ones are slowed by exactly $s$ (PI's guarantee); the ramp and
fast ones made at least one full turn in training, so every angle is familiar.

::video[clocks-by-parts]

Compare all three methods on Llama 2's 64 clocks. Watch the red bars under NTK-aware, and how the
fastest clock's step shrinks under PI:

::figure[yarn-frequencies]

**A deeper reason, found a year later.** The 2024 study *"Round and Round We Go!"* (summarised in
the [RoPE walkthrough's update](/papers/rope)) found that trained models use the **fast clocks for
position** ("attend to the previous word") and the **slow clocks as near position-free channels
for meaning**. NTK-by-parts protects both: the fast clocks keep their sharpness, and the slow
clocks never leave the angles their meaning-matching was trained on.

**Why it mattered.** This is the heart of YaRN. The insight isn't a formula; it's the question
"how many turns did this clock make in training?", asked separately for every clock.

## 3.3 · YaRN: by-parts plus a sharper softmax

> **The paper says:** *"…introducing a temperature $t$ on the logits before the attention softmax has
> a uniform impact on perplexity regardless of the data sample and the token position over the
> extended context window."*

**In plain words.** One more problem remains, and it isn't about clocks. Attention is a softmax
over every token in the context. Make the context 16 times longer and each word has 16 times more
candidates to spread its attention over. Attention gets **flatter**: more spread out, less
decisive. The technical word is that its **entropy** rises.

The fix is old and simple: **sharpen the softmax** by dividing the scores (the logits) by a
**temperature** $t < 1$ before it:

$$\operatorname{softmax}\!\left(\frac{q_m^\top k_n}{t\,\sqrt{|D|}}\right),
\qquad \sqrt{\tfrac{1}{t}} = 0.1\,\ln s + 1 .$$

The formula for $t$ isn't derived; the authors **fitted** it, by finding the best temperature for
LLaMA models at many stretches. It worked on Llama 2 too, which hints it's somewhat universal. Here's
what it gives:

| Stretch $s$ | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| Logits multiplied by $1/t$ | 1.14 | 1.30 | 1.46 | 1.63 | 1.81 |

**Why a logarithm? A toy derivation.** The fitted formula grows like $\ln s$. Here's a simple model
that shows why that shape is natural. Suppose a query has one "right" key with score $a$ and $n$
other keys with score about 0. The right key's share of attention is

$$\frac{e^{a}}{e^{a} + n} .$$

Now make the context $s$ times longer: $n \to sn$. To keep the same share, the right key's score
must grow to $a'$ with $e^{a'} = s\,e^{a}$, that is $a' = a + \ln s$. If we get there by multiplying
every score by a factor $c$, then $c\,a = a + \ln s$, so

$$c = 1 + \frac{\ln s}{a} .$$

A sharpening factor of the form $1 + (\text{constant}) \times \ln s$: exactly the shape the authors
found by fitting. (Their $1/t = (0.1\ln s + 1)^2 \approx 1 + 0.2\ln s$ for moderate $s$, which in this
toy corresponds to a right-key lead of about 5.) The toy is only an intuition, and real attention
patterns are messier, but the logarithm comes from the same place: $s$ times more competitors
means a $\ln s$ bigger lead is needed to win the same share.

::video[sharper-softmax]

**A free implementation.** Changing attention code is a nuisance (it breaks fast kernels like
FlashAttention). But dividing $q^\top k$ by $t$ is the same as multiplying **both** $q$ and $k$ by
$\sqrt{1/t}$, and RoPE already multiplies $q$ and $k$ by cosines and sines. So YaRN just scales
the precomputed cos and sin tables by $\sqrt{1/t}$:

> *"…it has zero overhead during both inference and training, as rotary position embeddings are
> generated in advance and are reused for all forward passes."*

**YaRN = NTK-by-parts + this temperature.** Here are both, in NumPy:

```python
import numpy as np

def yarn_thetas(d=128, base=10000, L=4096, s=16, alpha=1, beta=32):
    """NTK-by-parts: slow clocks get θ/s, fast clocks keep θ, a straight-line mix in between."""
    theta = base ** (-np.arange(0, d, 2) / d)             # RoPE's d/2 clock speeds
    r = L * theta / (2 * np.pi)                           # full turns each clock makes in training
    gamma = np.clip((r - alpha) / (beta - alpha), 0, 1)   # 0 = slow clock, 1 = fast clock
    return (1 - gamma) * theta / s + gamma * theta

def yarn_qk_scale(s):
    """Multiply q and k by this, so every logit grows by its square (YaRN's temperature)."""
    return 0.1 * np.log(s) + 1

L, s = 4096, 16
theta = 10000 ** (-np.arange(0, 128, 2) / 128)
new = yarn_thetas(L=L, s=s)
print((new == theta).sum(), np.isclose(new, theta / s).sum(), round(yarn_qk_scale(s) ** 2, 3))  # 21 18 1.631
seen_all = theta * L >= 2 * np.pi                         # these clocks met every angle in training
assert np.all(seen_all | (new * L * s <= theta * L + 1e-9))  # no clock leaves its trained range
```

Use `new` in place of `theta` in the RoPE code from the [RoPE walkthrough](/papers/rope), multiply
the rotated queries and keys by `yarn_qk_scale(s)`, and that's YaRN.

**Why it mattered.** The temperature costs nothing and fixes a problem that clock-tuning can't
touch. Together, the two pieces beat every earlier method, with and without fine-tuning.

## 3.4 · Dynamic scaling: stretch only as far as you need

> **The paper says:** *"…it allows the model to gracefully degrade instead of immediately breaking
> when hitting the trained context limit $L'$."*

**In plain words.** A fixed stretch has two downsides. If you set $s = 16$, then even a short
100-token prompt gets its clocks slowed 16×, which costs a little quality for nothing. And past
$sL$ tokens the model hits a new wall.

**Dynamic scaling** picks the stretch on the fly, from the current length $l'$:

$$s = \max\!\left(1,\; \frac{l'}{L}\right).$$

Short texts get $s = 1$, plain RoPE. As the text grows past $L$, the clocks slow just enough. It
works **without any fine-tuning**: on LLaMA 7B (trained on 2,048 tokens), Dynamic-YaRN kept
perplexity low out to 32,768 tokens, where Dynamic-PI fell apart:

| Context | 2k | 4k | 8k | 16k | 32k |
|---|---|---|---|---|---|
| Dynamic PI | 4.05 | 3.90 | 6.18 | > 10 | > 100 |
| Dynamic YaRN | 4.05 | **3.67** | **3.65** | **3.33** | **3.45** |

(Perplexity: lower is better; around 3–4 is healthy, above 10 is broken.)

**One trap, with the KV cache.** During generation, a model stores past keys so it doesn't
recompute them (the [KV cache](/concepts/text-generation)). Normally keys are stored **already
rotated**, which is fine because a key's rotation never changes. With dynamic scaling it *does*
change: every time $s$ grows, every clock's speed changes, so every old key's rotation is now
wrong. The paper's advice: cache keys **before** rotating them, and rotate on the fly.

**Why it mattered.** Dynamic scaling is a free upgrade for any RoPE model, no training needed. It
shipped in popular open-source libraries and in Qwen's early long-context models.

## 4 · Experiments: 128k tokens for 400 steps of training

> **The paper says:** *"YaRN reaches state-of-the-art performances in context window extensions
> after fine-tuning on less than ~0.1% of the original pre-training data."*

**In plain words.** The recipe is short. Take Llama 2 (7B or 13B, trained on 4,096 tokens), apply
YaRN with $s = 16$, fine-tune for **400 steps** on long books (PG19, cut into 64k-token chunks).
For $s = 32$, train **200 more** steps, still on 64k-token chunks.

**"Train short, test long."** The $s = 32$ model never saw a text longer than 64k tokens in
fine-tuning, yet it reads 128k. Perplexity on 128k-token documents (Proof-pile; lower is better):

| Model | 8k | 32k | 64k | 128k |
|---|---|---|---|---|
| Llama 2 7B + YaRN, $s = 16$ | 3.51 | 2.65 | 2.42 | > 10 |
| Llama 2 7B + YaRN, $s = 32$ | 3.56 | 2.70 | 2.45 | **2.37** |
| Llama 2 13B + YaRN, $s = 32$ | 3.29 | 2.53 | 2.31 | **2.24** |

Perplexity keeps **falling** as the context grows: the model is actually using the extra text,
not just surviving it.

**Head to head.** The cleanest comparison is the paper's ablation: LLaMA 7B, stretched 16× to
32k, every method given the same 400 fine-tuning steps. Perplexity at 32k tokens:

| Method | Fine-tuned 400 steps | No fine-tuning |
|---|---|---|
| Position Interpolation | 3.57 | > 100 |
| NTK-aware | 8.49 | > 10 |
| NTK-by-parts | 2.81 | > 10 |
| **YaRN** | **2.77** | **3.45** |

Each step of the story shows up: NTK-aware's extrapolation hurts it after fine-tuning (8.49);
NTK-by-parts fixes that (2.81); the temperature adds a little more, and makes the model usable
even with no training at all.

**Finding a needle.** In the **passkey** test, a five-digit number is hidden at a random spot in
a long stretch of filler text and the model must repeat it. The 128k YaRN models (7B and 13B)
found it **99.4%** of the time across their full context.

**What it costs.** Short-text skills barely moved. On standard benchmarks, Llama 2 7B went from
53.1 → 52.1 (ARC-Challenge), 77.8 → 78.4 (HellaSwag), 43.8 → 41.7 (MMLU) and 39.0 → 37.3
(TruthfulQA) after stretching to 128k. MMLU dropped the most, about 2 points for 7B and 4 for 13B.
And training was cheap: about 128 A100-hours to reach 32k on LLaMA, against 640 for PI to reach
16k, and 6,400 for Code Llama's 100k.

**Why it mattered.** Long context went from "pretrain a new model" to "fine-tune for an afternoon
on a few GPUs". That put 100k-token open models within reach of small labs and hobbyists.

## 5 · Conclusion, and what happened next

> **The paper says:** *"…YaRN improves upon all existing RoPE interpolation methods and can act as a
> drop-in replacement to PI, with no downsides and minimal implementation effort."*

**In plain words.** The one-sentence summary: **to read $s$ times further, leave RoPE's fast clocks
alone, slow the slow clocks by $s$, blend the ones in between, and sharpen the softmax by a factor
that grows like $\ln s$.**

YaRN quickly became standard equipment:

- **Qwen 2.5** and **DeepSeek-V2/V3** use YaRN to stretch their context to 128k tokens.
- **Llama 3.1**'s own long-context scaling follows the same by-parts idea: leave short-wavelength
  clocks alone, slow long-wavelength clocks by a fixed factor, blend smoothly in between.
- **Raising the base** at pretraining time (Llama 3's 500,000) is the other half of the story:
  make the slow clocks slow enough from the start that long texts never push them off their
  trained slice. The [RoPE walkthrough's update](/papers/rope) explains why that protects the
  model's "meaning channels".

The bigger lesson is a way of thinking. RoPE looks like one mechanism, but it's really **64 clocks
with 64 different jobs**. YaRN works because it stops treating them as one.

**Build it yourself.** The [RoPE walkthrough](/papers/rope) has RoPE in a dozen lines of NumPy;
swap in `yarn_thetas` from section 3.3 and you have YaRN. The [transformer
track](/tracks/transformers) builds the rest of the model around it.
