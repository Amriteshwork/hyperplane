---
title: RoFormer (RoPE), explained
summary: The 2021 paper behind rotary position embeddings, the way Llama, Mistral, Qwen, Gemma and DeepSeek know where each word is. Every equation derived from scratch, from "what is a rotation?" to why far-away words fade, with the key ideas animated and three figures to play with, plus a 2024 update on what trained models actually do with each clock.
order: 3
source: su-rope
concepts: [vectors, matrix-multiplication, embeddings, self-attention, multi-head-attention, positional-encoding, transformer]
track: transformers
videos:
  stamp-vs-rotate:
    title: "Two ways to say “I'm at position t”: add a stamp, or rotate"
    youtube: T8S1RPHluOk
  turntable:
    title: Rotate each vector by its position, and only the gap survives
    youtube: _lErlvlqVIg
  spinning-arrows:
    title: Why far-away words whisper
    youtube: M7RxKjzC9bY
  previous-token-head:
    title: "A previous-token head: turn the query back one notch"
    youtube: _h647ojd3HM
---

Here's a puzzle. You have two arrows, a **query** and a **key**, and a number you get from
them: their **dot product**. The query sits at position $m$ in a sentence, the key at position
$n$. You're allowed to change each arrow using **only its own position**: the query can't see
$n$, the key can't see $m$. Your goal: make the dot product depend on **how far apart they are**,
$m - n$, and not on where they are.

Stop and think about it for a moment. It sounds impossible: each arrow knows only its own address,
yet the result has to know only their *difference*.

In April 2021, Jianlin Su and five colleagues at Zhuiyi Technology, a Chinese AI company, posted a
paper solving exactly this puzzle. (Su had first explained the idea on his Chinese-language blog a
month earlier.) The answer fits in four words: **rotate, don't add.** They called it
**rotary position embedding, RoPE**, and the model built on it **RoFormer**.

RoFormer itself was a modest BERT-sized model, and the paper's experiments were modest too. But
the idea spread fast. EleutherAI tested it and adopted it for GPT-J and GPT-NeoX; Google used it
in PaLM; Meta used it in LLaMA. Today almost every open large language model, Llama, Mistral,
Qwen, Gemma, DeepSeek, uses RoPE. If you've chatted with an open model, it has rotated your
words.

**How to read this page.** It follows the paper's sections, numbered as in the original, with one
addition: **section 0** is a toolkit that builds everything we need about rotations from nothing,
including complex numbers. If you're comfortable with rotation matrices and $e^{i\theta}$, skim
it. Every section has **the paper says** (a short quote), **in plain words**, the **maths derived
step by step**, and **why it mattered**. Derivations are in full: no "it can be shown that". Two
figures let you play with the ideas, and three animations show them moving. You need to know what
a dot product and a matrix are, and that attention scores are dot products of queries and keys.
The links under **Go deeper** cover those from scratch.

## Abstract and 1 · Where am I, and how far apart are we?

> **The paper says:** *"Specifically, the proposed RoPE encodes the absolute position with a
> rotation matrix and meanwhile incorporates the explicit relative position dependency in
> self-attention formulation."*

**In plain words.** A transformer's attention compares every word with every other word, and the
formula, $\operatorname{softmax}(QK^\top/\sqrt d)V$, has **no idea of order**. Shuffle the words
and you shuffle the outputs, nothing more. *"Dog bites man"* and *"man bites dog"* look like the
same bag of words. So we have to tell the model where each word is. There are two kinds of
"where":

- **Absolute position:** "I am word number 37." Like a house number.
- **Relative position:** "That word is 3 to my left." Like "the bakery is three doors down".

Language mostly runs on the **relative** kind. An adjective modifies the noun right after it,
whether the phrase is at word 5 or word 500. A verb looks for its subject a few words back. If
the model learns "look one word back" at position 5, it should get position 500 for free.

The abstract's sentence above sounds like a contradiction, and it's the whole paper: RoPE
**encodes the absolute position** (each word is changed by its own position number only) and yet
attention **sees only relative positions**. That's our opening puzzle, solved. The rest of this
page shows how, and proves it's essentially the only way.

**Why it mattered.** Before RoPE you had to choose: absolute schemes are simple but don't think in
distances; relative schemes think in distances but complicate attention (section 2.3). RoPE gives
both at once, for the price of a few multiplications.

**Go deeper:** [Positional encoding](/concepts/positional-encoding) ·
[Self-attention](/concepts/self-attention)

## 0 · A toolkit: three facts about turning arrows

This section isn't in the paper. It builds the three facts the paper takes for granted, from
nothing. Each one is short.

### Fact 1: a rotation is a matrix, and here's why

Take an arrow in the plane, $(x, y)$, and turn it anticlockwise by an angle $\alpha$. Where does it
land?

Start with the two simplest arrows. The arrow $(1, 0)$ points east; turned by $\alpha$, it points
at angle $\alpha$, so it lands at $(\cos\alpha, \sin\alpha)$. That's the definition of cosine and
sine: the coordinates of a point at angle $\alpha$ on a circle of radius 1. The arrow $(0, 1)$
points north, $90°$ further round; it lands $90°$ further round too, at
$(\cos(\alpha + 90°), \sin(\alpha + 90°)) = (-\sin\alpha, \cos\alpha)$.

Now any arrow is a mix of those two: $(x, y) = x\,(1, 0) + y\,(0, 1)$. Turning the whole plane
doesn't break mixes (a rotation is *linear*: turn two arrows and add them, or add them and turn
the sum, same result). So:

$$
\begin{aligned}
(x, y) \;\longmapsto\;& x\,(\cos\alpha, \sin\alpha) + y\,(-\sin\alpha, \cos\alpha) \\
=\;& (x\cos\alpha - y\sin\alpha,\;\; x\sin\alpha + y\cos\alpha).
\end{aligned}
$$

Written as a matrix, with the two landing spots as its columns:

$$R(\alpha) = \begin{pmatrix} \cos\alpha & -\sin\alpha \\ \sin\alpha & \cos\alpha \end{pmatrix},$$

$$R(\alpha)\begin{pmatrix} x \\ y \end{pmatrix}
= \begin{pmatrix} x\cos\alpha - y\sin\alpha \\ x\sin\alpha + y\cos\alpha \end{pmatrix}.$$

That's the only matrix in this paper. Everything else is this one, repeated.

### Fact 2: rotations add, undo, and never change lengths

Three common-sense facts about turning things, each provable in a line:

- **Turns add up.** Turning by $a$ and then by $b$ is turning by $a + b$:
  $R(b)R(a) = R(a + b)$. (Multiply the matrices and the angle-addition formulas
  $\cos(a+b) = \cos a\cos b - \sin a \sin b$, $\sin(a+b) = \sin a\cos b + \cos a\sin b$ pop out.)
- **The transpose undoes a turn.** Flip $R(a)$ across its diagonal:
  $R(a)^\top = \begin{pmatrix} \cos a & \sin a \\ -\sin a & \cos a \end{pmatrix}$. Since
  $\cos(-a) = \cos a$ and $\sin(-a) = -\sin a$, that's exactly $R(-a)$: turning back.
- **Lengths don't change.** Turning an arrow doesn't stretch it. Algebraically:
  $(x\cos\alpha - y\sin\alpha)^2 + (x\sin\alpha + y\cos\alpha)^2 = x^2 + y^2$, because the cross
  terms cancel and $\cos^2 + \sin^2 = 1$.

Put the first two together and you get **the identity that RoPE is built on**:

$$\boxed{\;R(a)^\top R(b) = R(-a)\,R(b) = R(b - a)\;}$$

Turn back by $a$, then forward by $b$: you've turned by $b - a$ in total. Remember this box. In
three sections it will solve our puzzle.

### Fact 3: complex numbers are just arrows that know how to turn

The paper does most of its maths with complex numbers, so let's build them from arrows. Write the
arrow $(x, y)$ as the single number $x + iy$. The symbol $i$ is a **label for the second axis**,
with one rule: $i \cdot i = -1$.

Why that rule? Watch what multiplying by $i$ does to an arrow:

$$i\,(x + iy) = ix + i^2 y = -y + ix ,$$

which, as an arrow, is $(x, y) \mapsto (-y, x)$.

$(1, 0)$ goes to $(0, 1)$; $(0, 1)$ goes to $(-1, 0)$. **Multiplying by $i$ is a $90°$ turn.** And
$i \cdot i = -1$ just says two quarter-turns make a half-turn, which flips an arrow round. Now
multiply by the number $\cos\alpha + i\sin\alpha$:

$$
\begin{aligned}
(x + iy)(\cos\alpha + i\sin\alpha) &= x\cos\alpha + i\,x\sin\alpha \\
&\quad + i\,y\cos\alpha + i^2 y\sin\alpha \\
&= (x\cos\alpha - y\sin\alpha) \\
&\quad + i\,(x\sin\alpha + y\cos\alpha).
\end{aligned}
$$

Compare with Fact 1: it's **exactly the rotation matrix**, in disguise. Multiplying by
$\cos\alpha + i\sin\alpha$ turns an arrow by $\alpha$. This number has a famous short name,
**Euler's formula**:

$$e^{i\alpha} = \cos\alpha + i\sin\alpha .$$

Why on earth is a turn an exponential? Here's the intuition. Think of $z(t) = e^{it}$ as a point
moving in time. Its velocity is $\frac{d}{dt}e^{it} = i\,e^{it} = i\,z$: always the current
position turned by $90°$. A point whose velocity is always at right angles to where it is, with the
same length, can't get closer to or further from the centre. It just goes **round a circle** of
radius 1, at speed 1. After time $\alpha$ it has travelled $\alpha$ around: it's at angle $\alpha$.
That's Euler's formula.

The exponential form makes Fact 2 free: $e^{ia}e^{ib} = e^{i(a+b)}$ is the ordinary exponent rule,
and it says *turns add up*.

One last piece. The **conjugate** $\overline{z}$ of $z = x + iy$ is $x - iy$: the mirror image,
same arrow turned the other way. For two arrows $q = a + ib$ and $k = c + id$:

$$q\,\overline{k} = (a + ib)(c - id) = (ac + bd) + i\,(bc - ad).$$

Look at the real part: $ac + bd$. That's **the dot product** of $(a, b)$ and $(c, d)$! So

$$\mathbf q \cdot \mathbf k = \operatorname{Re}\big[\,q\,\overline{k}\,\big].$$

In words: multiply one arrow by the mirror of the other, and the real part is their dot product.
(The imaginary part, $bc - ad$, measures how much they point *across* each other; we won't need
it.) This is the one formula from the toolkit the paper uses on almost every page.

**Go deeper:** [Vectors](/concepts/vectors) · [Matrix multiplication](/concepts/matrix-multiplication)

## 2.1 · The setup: the one place position can matter

> **The paper says:** *"The self-attention first incorporates position information to the word
> embeddings and transforms them into queries, keys, and value representations."*

**In plain words.** The paper starts by writing attention so that position could enter anywhere.
Word $m$ has an embedding $x_m$. It's turned into a query, key and value by functions that may
also see its position:

$$q_m = f_q(x_m, m), \quad k_n = f_k(x_n, n), \quad v_n = f_v(x_n, n).$$

Then attention does what it always does: score every pair, softmax, blend:

$$a_{m,n} = \frac{\exp\!\big(q_m^\top k_n / \sqrt d\big)}{\sum_{j}\exp\!\big(q_m^\top k_j / \sqrt d\big)},
\qquad o_m = \sum_n a_{m,n}\, v_n .$$

Here's the observation to hold on to. **Where words are only matters through the scores**
$q_m^\top k_n$: they decide who listens to whom. So the whole question "how should a transformer
know positions?" comes down to: **what should $f_q$ and $f_k$ be, so that the score $q_m^\top k_n$
knows what it should about positions?** Everything in the paper is an answer to that.

**Why it mattered.** Framing the problem as "design $f_q$ and $f_k$" is what makes a clean,
derivable answer possible. Earlier methods were mostly engineered tweaks; this framing turns it
into a maths problem with a solution.

**Go deeper:** [Self-attention](/concepts/self-attention) · [Softmax](/concepts/softmax)

## 2.2 · Absolute positions: adding a stamp, and the mess it makes

> **The paper says:** *"…a typical choice of Equation (1) is
> $f_{t:t\in\{q,k,v\}}(x_i, i) := W_{t:t\in\{q,k,v\}}(x_i + p_i)$."*

**In plain words.** The original transformer's answer, and GPT-2's and BERT's, is to **add a
position vector** $p_m$ to each word's embedding before anything else, like stamping each word
with its seat number. The original used sine waves for $p_m$; GPT-2 and BERT learned them.

Now let's look at what the attention score becomes. Expand it with the distributive law:

$$
\begin{aligned}
q_m^\top k_n &= \big(W_q(x_m + p_m)\big)^\top \big(W_k(x_n + p_n)\big) \\
&= x_m^\top W_q^\top W_k\, x_n && \text{word meets word} \\
&\quad + x_m^\top W_q^\top W_k\, p_n && \text{word meets place} \\
&\quad + p_m^\top W_q^\top W_k\, x_n && \text{place meets word} \\
&\quad + p_m^\top W_q^\top W_k\, p_n && \text{place meets place.}
\end{aligned}
$$

Four terms, and only the first one is about meaning. Notice three problems:

1. **Meaning and position are tangled.** The middle two terms compare a word's *meaning* with
   another word's *seat number*. "Does the word *cat* like seat 17?" is rarely a useful question,
   but the model pays for it in every score and has to learn to tune it out.
2. **Distance is only implicit.** Nothing in the formula says "$m - n$". For sinusoidal stamps,
   $p_m^\top p_n$ does depend only on $m - n$ (the clocks argument in the
   [attention paper](/papers/attention-is-all-you-need)), but the learned matrices $W_q^\top W_k$
   sit in between and can break that.
3. **Learned stamps stop at the training length.** If $p_m$ is a learned table with 512 rows, there
   *is* no $p_{600}$.

The animation shows the more basic trouble with adding: it changes the vector's **length and
direction** depending on where the word is. Rotating, as we'll see, only changes the direction.

::video[stamp-vs-rotate]

**Why it mattered.** This four-term expansion is the case for a different design. If position
entered the score as *one clean factor that depends on $m - n$*, the model would get distances
without the noise.

**Go deeper:** [Positional encoding](/concepts/positional-encoding) ·
[Embeddings](/concepts/embeddings)

## 2.3 · Relative positions: patching the score from the outside

> **The paper says** (of earlier methods, in its introduction): *"…thus render them unsuitable for
> the linear self-attention architecture."*

**In plain words.** Since 2018 a family of methods has gone straight for distance. They take the
four-term expansion above and replace the position parts with things that depend on $m - n$:

- **Shaw et al. (2018)** add a learned vector $r_{m-n}$ to the key, one per distance (clipped, so
  every distance beyond some maximum shares one vector).
- **Transformer-XL (2019)** rewrites all four terms with sinusoids of $m - n$ and extra learned
  vectors.
- **T5 (2020)** simply adds a learned number $b_{m-n}$ to each score: one bias per bucket of
  distances.

These work well. But they all edit **the table of scores itself**: to add $b_{m-n}$ you need the
score for each specific pair $(m, n)$, written out. That clashes with two things people want:

- **Linear attention** (section 3.3) never builds the $n \times n$ table, so there's nowhere to add
  the bias.
- **Simplicity.** Each method changes the attention code in its own way.

**Why it mattered.** The ideal would be relative like these, but applied to $q$ and $k$
**separately**, before they ever meet, so attention itself stays untouched. That's the opening
puzzle again, and section 3 answers it.

**Go deeper:** [Multi-head attention](/concepts/multi-head-attention)

## 3.1 · The wish: a score that only knows the gap

> **The paper says:** *"…we hope that the inner product encodes position information only in the
> relative form."*

**In plain words.** Here's the puzzle, written as an equation. Find functions $f_q$ and $f_k$ such
that

$$\big\langle f_q(x_m, m),\; f_k(x_n, n)\big\rangle = g(x_m, x_n, m - n)$$

for some function $g$. Read it slowly:

- The **left side** is the attention score. The query is built from word $m$ and **its own**
  position $m$; the key from word $n$ and **its own** position $n$.
- The **right side** says the score may depend on the two words, but on positions **only through
  the gap** $m - n$.

**Pause and predict.** Before reading on, look back at the box in section 0:
$R(a)^\top R(b) = R(b - a)$. The left side uses $a$ and $b$ separately; the right side only their
difference. Sound familiar?

Here's the geometric intuition. A dot product is $\|q\|\,\|k\|\cos(\text{angle between them})$:
it only cares about the lengths and the **angle between** the arrows. What can you do to two
arrows, each knowing only its own position, so that the angle between them changes by an amount
that depends only on the gap?

Picture two people standing on a playground **turntable**. Spin the turntable and both of them
turn, but the angle between them (as seen from the centre) never changes. Now the trick: **turn
each arrow by an angle proportional to its position.** The query turns by $m\theta$, the key by
$n\theta$. The angle between them changes by $(n - m)\theta$. Move both words 10 places further
along the sentence and both turn by $10\theta$ more: same turntable, same angle between them,
same score.

**Why it mattered.** Every relative scheme before had to *break into* attention to inject
distance. Turning the arrows lets the dot product, the thing attention computes anyway, **do the
subtraction for you**.

## 3.2.1 · The 2D case: rotate the query and the key

> **The paper says:** *"…simply rotate the affine-transformed word embedding vector by amount of
> angle multiples of its position index…"*

**In plain words.** Start with the smallest case: embeddings with just **2 numbers**, so queries
and keys are arrows in the plane. RoPE says: compute the query and key as usual, then **rotate**
them by their position:

$$q_m = R(m\theta)\, W_q x_m, \qquad k_n = R(n\theta)\, W_k x_n ,$$

where $\theta$ is a fixed angle per position, like $20°$. Word 0 isn't turned at all, word 1 turns
$20°$, word 2 turns $40°$, and so on.

**The proof, in one line.** Write $\mathbf q = W_q x_m$ and $\mathbf k = W_k x_n$ for the plain,
unrotated query and key. The transpose of a product is the product of transposes in reverse,
$(AB)^\top = B^\top A^\top$, so:

$$
\begin{aligned}
q_m^\top k_n &= \big(R(m\theta)\,\mathbf q\big)^\top \big(R(n\theta)\,\mathbf k\big) \\
&= \mathbf q^\top\, R(m\theta)^\top R(n\theta)\; \mathbf k \\
&= \mathbf q^\top\, R\big((n - m)\theta\big)\; \mathbf k . \qquad \text{(the box from section 0)}
\end{aligned}
$$

The positions $m$ and $n$ have vanished. All that's left is their **gap**, $n - m$. The puzzle is
solved.

**A worked example.** Take $\theta = 20°$ and the simplest query and key, both $(1, 0)$, pointing
east. Put the query at $m = 2$ and the key at $n = 5$:

- the query turns $2 \times 20° = 40°$; the key turns $5 \times 20° = 100°$;
- the angle between them is $60°$, so the score is $\cos 60° = 0.5$.

Now slide both 8 places along, to $m = 10$, $n = 13$: the query turns $200°$, the key $260°$. Wildly
different directions from before, but the angle between them is still $60°$, and the score is
still $0.5$. Move the key one more place, to $n = 14$, and the angle becomes $80°$: the score
drops to $\cos 80° \approx 0.17$.

::video[turntable]

**The same thing with complex numbers.** The paper writes it in the language of section 0's
Fact 3. Treat the 2D query and key as complex numbers; rotating by $m\theta$ is multiplying by
$e^{im\theta}$:

$$
\begin{aligned}
f_q(x_m, m) &= (W_q x_m)\, e^{im\theta}, \\
f_k(x_n, n) &= (W_k x_n)\, e^{in\theta}.
\end{aligned}
$$

The dot product is the real part of one times the mirror of the other. The mirror of
$e^{in\theta}$ is $e^{-in\theta}$ (the same turn, backwards), so:

$$
\begin{aligned}
q_m \cdot k_n &= \operatorname{Re}\Big[(W_q x_m)\,e^{im\theta}\;\overline{(W_k x_n)}\,e^{-in\theta}\Big] \\
&= \operatorname{Re}\Big[(W_q x_m)\,\overline{(W_k x_n)}\; e^{i(m - n)\theta}\Big].
\end{aligned}
$$

That's the paper's equation (12), the $g(x_m, x_n, m - n)$ we wished for. Two exponentials, and
the exponent rule subtracts the positions for us.

Now try it yourself. Shift both positions, and watch the bars in the strip stay level. Then switch
to **add a stamp**, the section 2.2 approach, and shift again:

::figure[rope-rotation]

**Why it mattered.** The proof is a single application of "turn back by $a$, then forward by $b$".
That's why RoPE is so easy to trust: the relative-position property isn't learned or approximate,
it's **exact**, for every query, every key and every position.

**Go deeper:** [Matrix multiplication](/concepts/matrix-multiplication) ·
[Vectors](/concepts/vectors)

## 3.4.1 · Is rotating the only answer? Deriving it from scratch

> **The paper says:** after a page of algebra in its section 3.4.1, that the wish has *"a
> straightforward solution"*. It's rotation.

**In plain words.** Section 3.2 *guessed* rotation and checked that it works. Section 3.4.1 of the
paper does something more satisfying: it starts from the wish, makes no guess, and shows that
the wish **forces** rotation. If you only read one derivation on this page, read this one. It's
four short steps.

**The setup.** Work in 2D, with complex numbers. To keep the notation light, write $q$ for the
plain query $W_q x_m$ and $k$ for the plain key $W_k x_n$. We want functions with

$$f_q(q, m)\;\overline{f_k(k, n)} = g(q, k, m - n) \tag{wish}$$

(the paper asks for the whole complex product to depend on the gap, not just its real part; it's
a slightly stronger wish that makes the maths tidy), and one sensible **starting condition**:
position 0 changes nothing,

$$f_q(q, 0) = q, \qquad f_k(k, 0) = k. \tag{start}$$

Every complex number is a **length** and an **angle**: $z = r\,e^{i\phi}$. So write each function
that way:

$$
\begin{aligned}
f_q(q, m) &= R_q(q, m)\, e^{i\Theta_q(q, m)}, \\
f_k(k, n) &= R_k(k, n)\, e^{i\Theta_k(k, n)}, \\
g &= R_g\, e^{i\Theta_g}.
\end{aligned}
$$

Here $R$ is a length and $\Theta$ an angle. And write the plain query and key as
$q = \|q\|\,e^{i\theta_q}$, $k = \|k\|\,e^{i\theta_k}$: lengths $\|q\|, \|k\|$, angles
$\theta_q, \theta_k$.

When you multiply complex numbers, **lengths multiply and angles add**; the mirror flips the sign
of the angle. So the wish splits into two separate equations, one for lengths and one for angles:

$$
\begin{aligned}
R_q(q, m)\, R_k(k, n) &= R_g(q, k, m - n), && \text{lengths} \\
\Theta_q(q, m) - \Theta_k(k, n) &= \Theta_g(q, k, m - n). && \text{angles}
\end{aligned}
$$

**Step 1: lengths don't change.** Put the query and key at the **same** position, $n = m$. The gap
is 0, so

$$R_q(q, m)\, R_k(k, m) = R_g(q, k, 0).$$

The right side doesn't depend on $m$, so we may as well evaluate it at $m = 0$, where the start
condition says nothing has changed: $R_g(q, k, 0) = \|q\|\,\|k\|$. The simplest solution, and the
one the paper takes, is that **position never changes the length**:

$$R_q(q, m) = \|q\|, \qquad R_k(k, n) = \|k\|.$$

Position, then, can only change the **angle**. That's already most of the way to "rotation".

**Step 2: every word gets the same extra angle.** Same trick for angles. With $n = m$:

$$\Theta_q(q, m) - \Theta_k(k, m) = \Theta_g(q, k, 0) = \theta_q - \theta_k ,$$

where the last step evaluated at $m = 0$ again. Rearrange so each side has one vector:

$$\underbrace{\Theta_q(q, m) - \theta_q}_{\text{depends on } q \text{ and } m}
= \underbrace{\Theta_k(k, m) - \theta_k}_{\text{depends on } k \text{ and } m}.$$

This has to hold for **every** query and **every** key. The left side can't secretly depend on $q$:
if it did, changing $q$ would change it while the right side stayed put. Likewise for $k$. So both
sides are a function of the position alone. Call it $\varphi(m)$: the **extra angle added at
position $m$**, the same for every word, query or key:

$$\Theta_q(q, m) = \theta_q + \varphi(m), \quad \Theta_k(k, n) = \theta_k + \varphi(n).$$

**Step 3: the extra angle grows in equal steps.** Now put the key **one place** behind the query,
$n = m - 1$, so the gap is 1. The angle equation says

$$\big(\theta_q + \varphi(m)\big) - \big(\theta_k + \varphi(m - 1)\big) = \Theta_g(q, k, 1),$$

$$\varphi(m) - \varphi(m - 1) = \Theta_g(q, k, 1) - \theta_q + \theta_k .$$

Look at the right side: it has **no $m$ in it**. So the step from each position to the next is
the same size, wherever you are. Call that step $\theta$. A staircase whose steps are all the same
height is a straight line: $\varphi(m) = \varphi(0) + m\theta$, and $\varphi(0) = 0$ by the start
condition. So

$$\varphi(m) = m\,\theta .$$

**Step 4: put it together.** The query keeps its length and gains angle $m\theta$:

$$f_q(q, m) = \|q\|\, e^{i(\theta_q + m\theta)} = \underbrace{\|q\|\,e^{i\theta_q}}_{q}\; e^{im\theta}
= q\, e^{im\theta},$$

and the same for the key. That is **rotation by $m\theta$**, exactly what section 3.2 guessed. Not
one possible answer; the answer that the wish, plus "position 0 changes nothing", leaves you.

**Why it mattered.** Most positional schemes are engineering choices you could argue about. This
one is **derived**: once you decide that scores should see only gaps, rotation by an angle
proportional to position is what you get. The only freedom left is the step $\theta$, and the
next section uses that freedom well.

## 3.2.2 · From 2D to d dimensions: a row of clocks

> **The paper says:** *"…we divide the d-dimension space into d/2 sub-spaces…"*

**In plain words.** Real queries and keys have $d = 64$ or $128$ numbers per head, not 2. The fix
is beautifully simple: **pair them up**. Numbers 1 and 2 form one arrow, 3 and 4 another, and so
on: $d/2$ little arrows. Rotate each pair by its position, each pair with **its own speed**
$\theta_i$:

$$R_{\Theta, m} = \begin{pmatrix}
R(m\theta_1) & & & \\
& R(m\theta_2) & & \\
& & \ddots & \\
& & & R(m\theta_{d/2})
\end{pmatrix},$$

$$\theta_i = 10000^{-2(i-1)/d}.$$

Blank entries are zeros: each $2 \times 2$ block turns its own pair and ignores the rest. A dot
product is a sum over coordinates, so it's a sum over pairs, and the 2D proof works pair by pair:

$$
\begin{aligned}
q_m^\top k_n &= \mathbf q^\top R_{\Theta, m}^\top R_{\Theta, n}\, \mathbf k
= \mathbf q^\top R_{\Theta, n - m}\, \mathbf k \\
&= \sum_{i=1}^{d/2} \mathbf q_{[i]}^\top\, R\big((n - m)\theta_i\big)\, \mathbf k_{[i]} ,
\end{aligned}
$$

where $\mathbf q_{[i]}$ is the $i$-th pair. Still only the gap. Still exact.

**Why many speeds and not one?** Imagine every pair turned at $\theta = 1$ radian per word. An
arrow that turns 1 radian per step comes back round after $2\pi \approx 6.28$ words. So distance 0
and distance 63 look almost the same (63 is almost exactly ten full turns). One clock can't tell
"next to me" from "63 words away": the angle **wraps around**.

The fix is the one your wall clock uses. The second hand is fast and tells moments apart within a
minute; the minute hand is slower; the hour hand slower still. Read together, they're never
ambiguous within a day. RoPE's speeds $\theta_i$ go from **1 radian per word** for the first pair
down to about **1/10,000** for the last, spaced evenly on a log scale. With $d = 128$:

| Pair | Speed $\theta_i$ (radians per word) | One full turn every… |
|---|---|---|
| 1 (fastest) | 1 | 6 words |
| 17 | 0.1 | 63 words |
| 33 | 0.01 | 630 words |
| 49 | 0.001 | 6,300 words |
| 64 (slowest) | 0.00012 | 54,000 words |

Fast pairs see fine detail ("is it the very next word?"); slow pairs see the big picture ("is it
in this paragraph or the last?").

These are **exactly the frequencies of the original transformer's sinusoidal encoding**. RoPE
keeps those clocks; it just uses them differently. Instead of writing the clock readings into the
embedding once (adding), it **turns the query and key by them**, in every attention layer
(multiplying).

> *"…which ensures stability during the process of encoding position information."*

That's Fact 2 at work: $R_{\Theta, m}$ is a rotation, so it **never changes a vector's length**.
Position can't blow up or shrink a query, however far into the text it is.

**Why it mattered.** The pairing trick turns a 2D idea into a $d$-dimensional one with zero new
theory, and the many speeds give every distance from 1 to tens of thousands of words its own
recognisable signature.

**Go deeper:** [Positional encoding](/concepts/positional-encoding) (the clocks, with a figure)

## 3.4.2 · Computing it fast: no big matrix needed

> **The paper says:** *"Taking the advantage of the sparsity of $R^d_{\Theta,m}$…"*

**In plain words.** $R_{\Theta, m}$ is a $d \times d$ matrix, but almost all of it is zeros.
Multiplying by it would waste $d^2$ operations on what is really $d/2$ small rotations. Write out
what a $2 \times 2$ rotation does to one pair $(x_1, x_2)$:

$$
\begin{aligned}
x_1 &\mapsto x_1\cos m\theta - x_2 \sin m\theta, \\
x_2 &\mapsto x_2 \cos m\theta + x_1 \sin m\theta.
\end{aligned}
$$

Each output is "the number times cos, plus its partner times sin" (with a minus sign on the first
partner). So for the whole vector:

$$R_{\Theta, m}\,x =
\begin{pmatrix} x_1 \\ x_2 \\ x_3 \\ x_4 \\ \vdots \end{pmatrix} \odot
\begin{pmatrix} \cos m\theta_1 \\ \cos m\theta_1 \\ \cos m\theta_2 \\ \cos m\theta_2 \\ \vdots \end{pmatrix}
+
\begin{pmatrix} -x_2 \\ x_1 \\ -x_4 \\ x_3 \\ \vdots \end{pmatrix} \odot
\begin{pmatrix} \sin m\theta_1 \\ \sin m\theta_1 \\ \sin m\theta_2 \\ \sin m\theta_2 \\ \vdots \end{pmatrix}$$

where $\odot$ means multiply element by element. Two element-wise multiplications and an
addition: about as cheap as adding a position stamp. The cosines and sines depend only on
positions, so they're computed once and reused by every layer and every head.

Here it is in NumPy, with the two properties checked:

```python
import numpy as np

def rope(x, base=10000):
    """Rotate each (even, odd) pair of x's columns; row m by angle m·θᵢ."""
    T, d = x.shape
    theta = base ** (-np.arange(0, d, 2) / d)       # d/2 speeds: 1, then slower and slower
    angle = np.arange(T)[:, None] * theta           # (T, d/2): the angle m·θᵢ
    cos, sin = np.cos(angle), np.sin(angle)
    x1, x2 = x[:, 0::2], x[:, 1::2]                 # first and second number of every pair
    out = np.empty_like(x)
    out[:, 0::2] = x1 * cos - x2 * sin              # the 2×2 rotation, written out
    out[:, 1::2] = x1 * sin + x2 * cos
    return out

rng = np.random.default_rng(0)
q, k = rng.normal(size=64), rng.normal(size=64)
Q = rope(np.tile(q, (50, 1)))                       # the same query, placed at positions 0…49
K = rope(np.tile(k, (50, 1)))                       # the same key, likewise
print(Q[7] @ K[3], Q[40] @ K[36], Q[7] @ K[2])      # -9.188  -9.188  -9.991
assert np.isclose(Q[7] @ K[3], Q[40] @ K[36])       # gap 4 both times: same score
assert np.allclose(np.linalg.norm(Q, axis=1), np.linalg.norm(q))  # lengths untouched
```

Three practical details the paper doesn't dwell on:

- **Only queries and keys are rotated, never values.** Position decides *who listens to whom*
  (the scores); the values are *what gets said*, and they're left alone.
- **It plays perfectly with the KV cache.** When a chatbot generates text it stores each past
  word's key (the [KV cache](/concepts/text-generation)). A key's rotation depends only on its own
  position, which never changes, so it's rotated once and stored. Relative-bias methods, by
  contrast, need the gap to every new query, which changes at each step.
- **Which numbers get paired doesn't matter.** Many popular implementations (Hugging Face's Llama
  code, GPT-NeoX) pair number $j$ with number $j + d/2$ instead of with its neighbour: the
  "rotate half" convention. It's the same idea with the dimensions listed in a different order.
  Since $W_q$ and $W_k$ are learned, the model doesn't care, but loading weights trained with one
  convention into code written for the other silently breaks a model. A classic bug.

**Why it mattered.** RoPE costs a few element-wise operations per layer and needs no change at all
to attention itself. That made it trivial to drop into existing code, which is a big part of why
it spread so fast.

## 3.3 and 3.4.3 · Long-term decay: far-away words whisper

> **The paper says:** *"…which means the inner-product will decay when the relative position
> increase."*

**In plain words.** Here's a second gift that came free with the rotations. Write the score in
complex form, one term per pair. Let $h_i = \mathbf q_{[i]}\,\overline{\mathbf k_{[i]}}$ be pair
$i$'s contribution before any rotation, and $\Delta = m - n$ the gap:

$$q_m^\top k_n = \operatorname{Re}\Big[\sum_{i=0}^{d/2-1} h_i\, e^{i\Delta\theta_i}\Big].$$

(Here the $i$ in the exponent is $\sqrt{-1}$, and $i$ under the sum is the pair's index. The paper
uses the same letter for both; sorry.)

Picture each term as an **arrow**. At $\Delta = 0$, every exponential is 1 and the arrows are just
the $h_i$. For a query and key that match well, the $h_i$ point roughly the same way, so laid tip
to tail they make a long chain: a big score. Now move the two words apart. Each arrow turns by
$\Delta\theta_i$, **each at its own speed**. They soon point every which way, and a chain of arrows
pointing in random directions curls up on itself: the sum **cancels**. That's the decay.

::video[spinning-arrows]

**The derivation (summation by parts).** The paper turns that picture into a bound. The trick is
the discrete cousin of integration by parts, called **Abel summation**. Define the running sums of
the spinning arrows, with no $h$ at all:

$$S_j = \sum_{i=0}^{j-1} e^{i\Delta\theta_i}, \qquad S_0 = 0 .$$

Each arrow is the difference of two running sums: $e^{i\Delta\theta_i} = S_{i+1} - S_i$. Substitute,
and set $h_{d/2} = 0$ for tidiness:

$$\sum_{i=0}^{d/2-1} h_i\,(S_{i+1} - S_i) = -\sum_{i=0}^{d/2-1} S_{i+1}\,(h_{i+1} - h_i).$$

Why is that true? Expand both sides and regroup by $S$. Each $S_{i+1}$ appears on the left twice:
once as $+h_i S_{i+1}$ and once (from the next term) as $-h_{i+1} S_{i+1}$. That's
$-S_{i+1}(h_{i+1} - h_i)$, the right side. The two boundary leftovers are $h_0 S_0$ and
$h_{d/2} S_{d/2}$, both zero by choice. Just as integration by parts moves the derivative from
one factor to the other, this moves the "difference" from the spinning arrows onto the $h$'s.

Now take sizes. The size of a sum is at most the sum of the sizes, and each $|h_{i+1} - h_i|$ is
at most the biggest one:

$$\Big|\sum_i h_i\, e^{i\Delta\theta_i}\Big| \;\le\; \underbrace{\Big(\max_i |h_{i+1} - h_i|\Big)}_{\text{about the words}}\;
\underbrace{\sum_{i=0}^{d/2-1} |S_{i+1}|}_{\text{about the distance}} .$$

This splits the score's ceiling into two factors: one depends only on the **content** (how much
neighbouring pairs of $q$ and $k$ differ), the other only on the **distance** $\Delta$. And the
distance factor, the average size of those running sums of spinning arrows, shrinks as $\Delta$
grows, because the arrows scatter and cancel. The paper shows this as a plot (its Figure 2); you
can draw it yourself:

::figure[rope-decay]

**An honest caveat.** This is a **ceiling**, not a promise. A trained model can still put lots of
attention on a word 1,000 places back if the query and key are built to line up through the slow
pairs, and models do exactly that when they need to. Notice too that the figure's solid line
assumes the query and key **already match**. A 2024 study went further and argued that decay
isn't the main reason RoPE works at all. That story is the **update at the end of this page**.

**Why it mattered.** The decay gives RoPE a sensible default ("nearby words matter more"), without
forbidding long-range attention. And the base, 10,000, controls how fast it happens: a knob that
became central once people wanted much longer contexts (the epilogue below).

## 3.3 · RoPE with linear attention

> **The paper says:** *"Since RoPE injects position information by rotation, which keeps the norm
> of hidden representations unchanged…"*

**In plain words.** Ordinary attention builds the full $n \times n$ table of scores, which is why
its cost grows with $n^2$. **Linear attention** (Katharopoulos et al., 2020; Performer, 2020)
replaces $\exp(q^\top k)$ with $\phi(q)^\top \phi(k)$ for some feature map $\phi$ that keeps
everything positive. Then the matrix product can be regrouped:

$$o_m = \frac{\sum_n \phi(q_m)^\top \phi(k_n)\, v_n}{\sum_n \phi(q_m)^\top \phi(k_n)}
= \frac{\phi(q_m)^\top \sum_n \phi(k_n)\, v_n^\top}{\phi(q_m)^\top \sum_n \phi(k_n)} .$$

The sums over $n$ don't depend on $m$: compute them once and every query reuses them. Cost grows
with $n$, not $n^2$. But the table of scores never exists, so there's nowhere to add a T5-style
bias $b_{m-n}$.

RoPE doesn't need the table. It rotates $\phi(q_m)$ by $m$ and $\phi(k_n)$ by $n$ separately, and
the regrouping still works. One wrinkle: rotated features can be negative, so the denominator
could be zero. The paper keeps the denominator **unrotated** (*"to avoid the risk of dividing
zero"*), which keeps it positive. (So the
weights no longer add up to exactly 1, a compromise the paper accepts.)

**Why it mattered.** It showed RoPE isn't tied to standard attention: any attention that computes
dot products of queries and keys can use it. That includes the fast, approximate kinds people
reach for with very long inputs.

**Go deeper:** [Self-attention](/concepts/self-attention)

## 4 · Experiments: modest numbers, big consequences

> **The paper says:** *"…RoFormer can significantly outperform BERT in three out of six
> datasets…"*

**In plain words.** The paper tests RoPE five ways. Honestly reported, the results are good but
not dramatic:

| Test | Compared with | Result |
|---|---|---|
| English→German translation (WMT 2014) | Transformer-base | BLEU 27.3 → **27.5** |
| Pretraining a BERT-sized model | BERT | masked-word loss falls **faster** |
| Fine-tuning on GLUE (6 tasks) | BERT | better on 3 (MRPC, STS-B, QQP), worse on 3 |
| Linear attention (Performer, enwik8) | Performer without RoPE | trains **faster**, lower loss |
| Chinese legal-case matching (CAIL2019-SCM) | WoBERT, 512 tokens | 68.10% → **69.79%** with 1,024 tokens |

The last row is the most telling. The authors trained a Chinese RoFormer, then fine-tuned it on
pairs of long legal documents. At 512 tokens it roughly matched its rivals. Allowed **1,024**
tokens, which RoPE handles without any new learned table, it pulled ahead. Long text is where
relative positions should help, and they did.

The paper is also candid about what's missing (section 4.5.5): no thorough explanation of *why*
it converges faster than other schemes, and no faithful explanation of why it does better on long
texts, even though other schemes share the long-term decay.

**The real evidence came afterwards.** A few months later, EleutherAI compared RoPE head to head
with learned absolute positions and T5's relative bias, in GPT-style models. At 1.5 billion
parameters on the Pile, validation loss went from 2.240 (learned) and 2.223 (T5) to **2.173**
(RoPE), and RoPE reached a given loss roughly **30% faster** than learned positions. They adopted it
for GPT-J and GPT-NeoX, and the rest of the field followed.

**Why it mattered.** A small paper with small tables became one of the most-used ideas in modern
AI. The lesson: a clean, derived idea that's trivial to implement can beat flashier methods, once
someone tests it at scale.

## 5 · Conclusion, and what happened next: stretching the clocks

> **The paper says:** *"…relative position can be naturally formulated using vector production in
> self-attention…"*

**In plain words.** The one-sentence summary: **rotate each query and key by an angle proportional
to its position; the dot product then turns absolute positions into relative ones by itself.**
Everything else, the many clock speeds, the cheap element-wise version, the decay, follows from
that.

The story didn't stop there, because RoPE's clocks gave people a **knob to turn**. A model trained
on 4,000-word texts has only ever seen its slow clocks turn through small angles. Ask it about
word 30,000 and the slowest pairs show angles it has never met; quality collapses. Several fixes
followed, all of them about the clocks:

- **Position interpolation** (Chen et al., Meta, 2023): squeeze the positions. To fit 32,000 words
  into a model trained on 2,048, divide every position by 16, so all angles stay in the familiar
  range. A short fine-tune and LLaMA handled 32,000-word inputs.
- **Raise the base** ("NTK-aware" scaling, 2023): instead of slowing every clock equally, change
  10,000 to something bigger. That slows the slow clocks a lot and the fast ones barely, so
  word-to-word detail survives. Code Llama used a base of 1,000,000; Llama 3 uses **500,000**. Try
  that in the decay figure above: the curve holds up much further out.
- **YaRN** (Peng et al., 2023): treat each pair differently, leaving fast clocks alone, stretching
  slow ones, and adjusting the softmax temperature to match. It has its own walkthrough:
  [YaRN, explained](/papers/yarn).

And RoPE escaped text. A picture has two directions, so vision models give half of each query's
pairs the **row** position and half the **column** position; video models such as Qwen2-VL add a
third set for **time**. The turntable just gets more axes.

**Build it yourself.** This site's [positional encoding](/concepts/positional-encoding) page shows
the sinusoidal clocks RoPE grew from, with RoPE itself in eight lines of PyTorch. The
[transformer track](/tracks/transformers) builds a small GPT from a NumPy array upward, so you can
see exactly where those eight lines go.

## Update (2024) · Round and Round: what trained models really do with the clocks

> **The 2024 paper says:** *"A common belief is that RoPE is useful because it helps to decay token
> dependency as relative distance increases. In this work, we argue that this is unlikely to be the
> core reason."*

Three years after RoFormer, a team at Google DeepMind and Oxford (Federico Barbero, Alex
Vitvitskyi, Christos Perivolaropoulos, Razvan Pascanu and Petar Veličković) asked a question the
original paper couldn't answer: **once a big model is trained, what does it actually do with each
of RoPE's clocks?** Their paper, *"Round and Round We Go! What makes Rotary Positional Encodings
useful?"* (ICLR 2025), opened up Gemma 7B (and checked Llama 3.1 8B) and found that the clocks
split into two jobs. It's the best "sequel" to RoFormer there is, so here are its insights, in the
same style as the rest of this page.

### Insight 1: the decay only happens for vectors that already match

Look back at section 3.4.3. The solid line in the decay figure is a query and key that are
**identical**: at distance 0 every pair lines up, and moving apart can only make things worse. But
real queries and keys are rarely identical. Two simple calculations show that the decay is not a
general law.

**A misaligned pair gets *closer* with distance.** Take one pair turning at speed $\theta$, with the
query pointing east and the key pointing south, $90°$ apart. As the key's position moves $\Delta$
places after the query's, RoPE turns it $\Delta\theta$ back towards east, so their score is
$\cos(90° - \Delta\theta)$: it **starts at 0 and rises** to its maximum at $\Delta = 90°/\theta$.
More generally (their Proposition 3.1), for any query and **any** distance $r$ you like, you can pick
a key that scores highest at exactly that distance: take the query and turn it back by $r$ notches,
$k = R(-r\theta)\,q$. Then at distance $r$, RoPE turns it forward again, $R(r\theta)\,k = q$, and the
two line up perfectly. RoPE doesn't force "near beats far"; it lets the model **choose** which
distance wins.

**Random vectors don't decay at all.** Suppose the query and key are random, every number drawn
independently from a bell curve centred on 0 (their Proposition 3.2). The score at distance $r$ is
$q^\top R_{\Theta, r}\, k = \sum_{a,b} q_a\, (R_{\Theta,r})_{ab}\, k_b$. Take the average. Because
$q$ and $k$ are independent, the average of each product splits, $\mathbb E[q_a k_b] =
\mathbb E[q_a]\,\mathbb E[k_b] = 0 \cdot 0$:

$$\mathbb E\big[q^\top R_{\Theta, r}\, k\big] = \sum_{a,b} \mathbb E[q_a]\,(R_{\Theta,r})_{ab}\,\mathbb E[k_b] = 0
\qquad \text{for every distance } r .$$

The average score is 0 at distance 1 and 0 at distance 1,000: no decay. The fading curve in
section 3.4.3 is real, but it's a property of **matching** vectors, not of RoPE in general.

### Insight 2: measure which clocks a model uses, with one inequality

How do you tell which clocks a trained model relies on? The score is a sum over pairs, and each
pair's contribution is a dot product of two little arrows. A dot product can never be bigger than
the two lengths multiplied (the **Cauchy–Schwarz inequality**: $|\mathbf a \cdot \mathbf b| =
\|\mathbf a\|\|\mathbf b\|\,|\cos\phi| \le \|\mathbf a\|\|\mathbf b\|$). So

$$\big|\,\mathbf q_{[i]} \cdot \mathbf k_{[i]}\big| \;\le\; \|\mathbf q_{[i]}\|\;\|\mathbf k_{[i]}\| .$$

If a model keeps pair $i$'s arrows **short**, that clock can't move the score much, whatever its
angle. Measuring the average length of each pair, layer by layer, is a cheap X-ray of which clocks
matter. The result in Gemma 7B (128 pairs per head):

> *"Gemma greatly prefers to use the lowest frequencies of RoPE…"*

Most of the length sits in the **slowest** clocks, in almost every layer. The fast clocks are mostly
switched off, except in the first and last layers. And the **values**, which RoPE never rotates,
show no such pattern, so it's RoPE's rotations that push the model there.

### Insight 3: the fastest clocks build "positional heads"

Some heads ignore meaning entirely and attend by position alone: a **diagonal** head (each word
looks at itself) and a **previous-token** head (each word looks one step back). In Gemma these
heads are exactly the ones that use the fastest clocks. The paper proves how (their Theorem 5.3),
and the construction is lovely:

- **Diagonal head.** Give every word the **same** query and key, $q = k = c$, living in the fastest
  pair. The score at distance $\Delta$ is $\|c\|^2\cos(\Delta \cdot 1\text{ rad})$: biggest at
  $\Delta = 0$, smaller everywhere else. Make $\|c\|$ large and softmax turns "biggest" into "all of
  the attention".
- **Previous-token head.** Same, but turn the query **back by one notch**: $q = R(-1\text{ rad})\,c$.
  Now the arrows line up at distance exactly 1.

(One fast clock alone has a weak spot: 44 radians is almost exactly 7 full turns, so distance 44
looks nearly like distance 0. Several fast clocks together fix it, because they don't all come back
round at the same time, the same wall-clock trick as in section 3.2.2.)

::video[previous-token-head]

Why the fastest clock? Because it's the most **sensitive**: it moves a full radian per word, so
"distance 0" and "distance 1" give very different angles. The slowest clock (base 10,000) needs
about 10,000 words to move one radian; it can't tell neighbours apart at all.

And here's the twist that makes RoPE essential. **Without any positional encoding** (called
**NoPE**), such a head is impossible (their Proposition 5.2). The proof is one example: feed in the
same token many times, "the the the the…". With no positions, every word has the identical query and
key, so every score is the same, and softmax spreads attention evenly: word $i$ gives itself only
$1/i$ of its attention. No sharp diagonal, no previous-token head. RoPE's fast clocks are what let a
word say "the one right before me", even among identical words.

### Insight 4: the slowest clocks carry meaning, and slowly drift

If the slow clocks barely turn, what are they for? Precisely that: **a pair that barely turns is
almost position-free**, so its dot product is almost a pure "does this key mean what I'm looking
for?" test. The paper found heads whose slow pairs act as **semantic channels**, like one head
whose words attend to the most recent apostrophe.

But "almost" is the problem. Over a long enough text, even the slowest clock turns noticeably, and
a match that was perfect at distance 10 is scrambled at distance 100,000. The paper proves that a
single rotating channel can't stay reliable over arbitrarily long contexts (their Theorem 6.1).

That explains a mystery from section 5 above. With base 10,000 (and 128 numbers per head), the
slowest clock turns once every ~54,000 words, so over Llama 3's 128,000-word context it would spin
**more than 2 full turns**, and a semantic channel would point the wrong way at times. With base
**500,000**, it turns once every ~2.6 million words: over 128,000 words it moves only about a
twentieth of a turn. **Raising the base keeps the meaning channels stable**, which, the paper
argues, is the real reason it helps long contexts.

Pick a clock and watch how its contribution changes with distance:

::figure[rope-bands]

### Insight 5: so why rotate the slowest clocks at all? p-RoPE

If the slowest clocks are best used as position-free channels, make them **exactly**
position-free: don't rotate them. **p-RoPE** keeps RoPE on only the fastest fraction $p$ of the
pairs and leaves the rest unrotated. $p = 1$ is ordinary RoPE; $p = 0$ is NoPE. They trained Gemma
2B models from scratch (8,000-token context) and measured perplexity (lower is better):

| Positions | Wiki | FlanV2 |
|---|---|---|
| NoPE (no positions) | 4.8594 | 6.6429 |
| RoPE, base 10,000 | 4.4627 | 6.4429 |
| RoPE, base 500,000 | 4.4485 | 6.4593 |
| 0.25-RoPE (rotate only the fastest quarter) | 4.5302 | 6.5111 |
| **0.75-RoPE** (leave the slowest quarter unrotated) | **4.4414** | **6.4422** |
| 0.75-RoPE "reversed" (drop the *fastest* quarter instead) | 4.4592 | 6.4683 |

Leaving the slowest quarter unrotated did as well as or slightly better than plain RoPE, and better
than raising the base; dropping the fastest quarter instead did worse. The gains are small, the
models are small, and the authors are clear that perplexity isn't the same as downstream skill. But
the picture is clean.

**Why it mattered.** RoFormer derived *what* RoPE is; this paper shows *how* models use it. RoPE is
less "a fading glow with distance" and more **a set of tools**: fast clocks for exact positions
("the word before me"), slow clocks for nearly position-free meaning. That reframing changes how
people stretch context too: the slow clocks are the fragile ones, which is exactly what
[YaRN](/papers/yarn) treats separately.
