---
title: Attention Is All You Need, explained
summary: The 2017 paper that introduced the transformer, section by section, in plain words. Attention, softmax, √d, multiple heads and the sine-wave positions are each derived from the problem they solve, with the key ideas animated.
order: 1
source: vaswani-attention-is-all-you-need
concepts: [softmax, embeddings, self-attention, causal-masking, multi-head-attention, positional-encoding, residual-connections, layer-normalization, adam-optimizer, transformer]
track: transformers
videos:
  sequential-vs-parallel:
    title: One word at a time versus all at once
    youtube: qPTDFBaE4CY
  encoder-decoder:
    title: "Translating a sentence: read everything, then write word by word"
    youtube: v_-BKyOL3q4
  qkv-lookup:
    title: "Attention as a soft dictionary lookup"
    youtube: R6f5Zrz-8F8
  multi-head-split:
    title: Eight small heads instead of one big one
    youtube: haNdqYzTeCI
  positional-clocks:
    title: Positions as a row of clocks
    youtube: xZ8tHYf5Xr8
---

In June 2017, eight researchers at Google published a 15-page paper about **machine
translation**: turning English sentences into German and French ones. It beat every system
before it, trained in a fraction of the time, and its central claim fit in the title:
you don't need the machinery everyone was using, **attention is enough**.

The architecture it introduced, the **transformer**, is now inside ChatGPT, Claude, Gemini
and Llama, inside image generators and protein-folding models. It's one of the most cited
papers in computer science.

**How to read this page.** Each section below follows one section of the paper, in order.
Each one starts with **what the paper says** (a short quote), then explains it **in plain
words**, shows it in a figure or animation, and says **why it mattered**. The key formulas
aren't just handed over: we start from the problem, try the obvious fixes, watch them fail,
and arrive at the paper's answer, so you could have invented it yourself. Links at the end
of each section go to this site's concept pages, which build each idea from scratch with
maths and runnable code. Keep the original open in another tab if you like. Section numbers
here match it.

## Abstract and 1 · The problem with reading one word at a time

> **The paper says:** *"We propose a new simple network architecture, the Transformer,
> based solely on attention mechanisms, dispensing with recurrence and convolutions
> entirely."*

**In plain words.** In 2017, the best translation systems were **recurrent neural networks**
(RNNs and their improved version, LSTMs). An RNN reads a sentence the way you'd read
through a keyhole: one word at a time, left to right, carrying a running summary in a
single vector, its "memory". To process word 50, it must first process words 1 to 49.

That causes two problems, and the introduction names both:

- **It's slow to train.** *"This inherently sequential nature precludes parallelization
  within training examples."* A GPU is thousands of small workers. An RNN can only give
  them one word at a time per sentence, so most of them sit idle.
- **It forgets.** Everything the network knows about word 1 has to survive being squeezed
  through 49 updates of the memory vector to reach word 50. Long-range connections fade.

The authors' idea: **drop the chain.** Let every word look directly at every other word,
all at once, using **attention**. Attention already existed. Since 2014, RNN translators
had used it as an add-on, to let the writer glance back at the input sentence. The
transformer makes attention the **only** way information moves.

::video[sequential-vs-parallel]

**Why it mattered.** "Everything at once" means a GPU can work on every position of a
sentence at the same time. That's what made it practical to train on vastly more text
than an RNN ever could, which is the road that leads to today's large language models.

**Go deeper:** [Recurrent neural networks](/concepts/recurrent-neural-networks) ·
[LSTM](/concepts/lstm) · [Self-attention](/concepts/self-attention)

## 2 · Background: shortcuts between distant words

> **The paper says:** *"In the Transformer this is reduced to a constant number of
> operations, albeit at the cost of reduced effective resolution due to averaging
> attention-weighted positions, an effect we counteract with Multi-Head Attention."*

**In plain words.** Others had tried to escape the RNN's chain too, mainly with
**convolutions**, which look at a small window of neighbouring words at a time. Stack enough
windows and distant words can eventually influence each other, but the signal still has to
hop through many layers to get there.

The paper's measure of this is the **path length**: how many steps information takes to
travel from one word to another. Through an RNN it's up to $n$ steps for a sentence of $n$
words. Through stacked convolutions it's fewer, but still grows with distance. Through
self-attention it's **one step**, between any two words, however far apart.

There's a catch, and the quote admits it: attention **blends**. If a word gathers
information from five others at once, it gets an average, and averages blur detail. Keep
that in mind; the fix, multi-head attention, is in section 3.2.2.

**Why it mattered.** Short paths are easier to learn. To use a word 40 positions back, a
transformer needs one well-placed attention weight, not a message that survives 40 relays.

**Go deeper:** [Residual connections](/concepts/residual-connections) (the other shortcut
the paper relies on)

## 3.1 · The big picture: an encoder and a decoder

> **The paper says:** *"The encoder maps an input sequence of symbol representations
> $(x_1, \ldots, x_n)$ to a sequence of continuous representations $z = (z_1, \ldots, z_n)$.
> Given $z$, the decoder then generates an output sequence $(y_1, \ldots, y_m)$ of symbols
> one element at a time."*

**In plain words.** Think of a human interpreter. First they **listen to the whole
sentence** and understand it. Then they **speak the translation, word by word**, glancing
back at what they heard whenever they need to. The transformer has one part for each job:

- **The encoder** reads the English sentence. Every word attends to every other word, in
  both directions, so "bank" can discover whether the sentence is about rivers or money.
  Out comes one rich vector per word.
- **The decoder** writes the German sentence, **one word at a time**. Each new word is
  chosen by looking at the German written so far, and at the encoder's vectors.

Each side is a stack of $N = 6$ identical **layers** (blocks). Every layer has the same two
sub-layers: **attention** (words exchange information) and a small **feed-forward network**
(each word processes what it gathered). Two small tricks wrap each sub-layer:

$$\text{output} = \operatorname{LayerNorm}\big(x + \operatorname{Sublayer}(x)\big)$$

- **$x + \ldots$, the residual connection.** The sub-layer only computes a *change* to add
  to $x$, rather than replacing it. Why does that help? Differentiate:
  $\frac{d}{dx}\big[x + F(x)\big] = 1 + F'(x)$. However small $F'$ is, the gradient passes
  through the **1** untouched. Without the $+x$, the gradient through a stack is a product of
  one factor per layer, and twelve factors of 0.5 make $0.5^{12} \approx 0.0002$: the
  early layers would barely learn. Information and gradients get an express lane.
- **LayerNorm** rescales each word's vector to a standard size, so values don't drift as
  layers stack up.

::video[encoder-decoder]

Here is one block, sized like GPT-2 and GPT-3 (which kept only the decoder side; more on that
at the end):

::figure[transformer-block]

**Why it mattered.** The whole model is the same block repeated. Making it more powerful
mostly means more blocks and wider vectors, which is exactly how models grew from this
paper's 65 million parameters (213 million for its big version) to hundreds of billions.

**Go deeper:** [Residual connections](/concepts/residual-connections) ·
[Layer normalization](/concepts/layer-normalization) · [Transformer](/concepts/transformer)

## 3.2.1 · Scaled dot-product attention

> **The paper says:** *"An attention function can be described as mapping a query and a
> set of key-value pairs to an output, where the query, keys, values, and output are all
> vectors."*

**In plain words.** This is the heart of the paper. Rather than start from the formula, let's
build it from the problem, and see that each piece is forced.

**The problem.** Take "it" in *"the animal didn't cross the street because it was too
tired"*. To understand "it", the model has to pull in information from "animal". In general:
**each word should collect information from the other words, mostly from the relevant
ones.** Say word $j$ comes in as a vector $x_j$. Word $i$'s output should be some combination
of what the other words have.

**First try: average everyone.** $o_i = \frac1n \sum_j x_j$. Every word gets the same
output, the sentence's average, so "it" learns nothing in particular about "animal". The
weights must depend on **relevance**.

**Second try: pick the single best match**, like a Python dictionary: bring a key, take the
entry that matches. But "pick the best" is a jump. Nudge a score a little and the choice
either doesn't change at all or flips completely. Training works by nudging weights and
measuring what improves (gradients), and a hard choice gives a gradient of zero almost
everywhere. Nothing would be learned.

**So: a soft choice.** Give every other word a weight $a_{ij}$: positive, adding up to 1,
bigger for better matches, and changing **smoothly** as the scores change. Then
$o_i = \sum_j a_{ij} \cdot (\text{what word } j \text{ hands over})$. Two things to design:
the **score**, and how scores become **weights**.

**The score: a dot product, but of what?** We need one number saying how well two words
match. The simplest is a dot product: large when two vectors point the same way, near 0 when
unrelated, negative when opposed. For a whole sentence it's a single matrix multiplication.
The obvious version, $x_i \cdot x_j$, has two flaws:

- **Every word matches itself best**, because $x_i \cdot x_i = \|x_i\|^2$. Words would
  mostly look at themselves.
- **It's symmetric**: $x_i \cdot x_j = x_j \cdot x_i$. Relevance isn't. "It" badly needs
  "animal"; "animal" barely needs "it".

Both go away if each word makes **two different vectors** with two learned matrices: a
**query** $q_i = x_i W^Q$, *what am I looking for?*, and a **key** $k_j = x_j W^K$, *what do I
offer?* The score $q_i \cdot k_j$ is no longer symmetric, and a word no longer matches itself
automatically. By the same logic, what a word is **matched on** needn't be what it **hands
over**, so a third matrix makes a **value** $v_j = x_j W^V$. That's a dictionary lookup,
made soft: queries, keys and values.

**The weights: deriving softmax.** Turn a row of scores $s_1, \ldots, s_n$ into weights.
Positive, whatever the score: exponentiate, since $e^{s} > 0$ always, and bigger scores
give bigger results. Adding up to 1: divide by the total.

$$a_j = \frac{e^{s_j}}{\sum_k e^{s_k}} .$$

That's **softmax**, and it's smooth everywhere. (Why not square the scores to make them
positive? Then a strongly *negative* score would count as a great match.) A bonus: add the
same number to every score and the $e^{c}$ cancels top and bottom, so only **differences**
between scores matter.

Stack every word's query, key and value as the rows of matrices $Q, K, V$, and all of the
above, for all words at once, is the paper's equation:

$$\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

Read it right to left, in three steps:

1. **$QK^\top$, score.** Each query is dot-multiplied with each key. A dot product is large
   when two vectors point the same way, so it measures *how well they match*. For $n$
   words that's an $n \times n$ table of scores: every word against every word.
2. **$\operatorname{softmax}$, turn scores into shares.** Each row becomes positive weights
   that add up to 1: "spend 70% of your attention here, 20% there…".
3. **$\times V$, blend.** Each word's output is its weighted average of everyone's values.

::video[qkv-lookup]

**And why divide by $\sqrt{d_k}$?**

> *"We suspect that for large values of $d_k$, the dot products grow large in magnitude,
> pushing the softmax function into regions where it has extremely small gradients."*

The paper only says "we suspect". We can derive it. At the start of training, think of each
number in a query and a key as independent noise with average 0 and spread (variance) 1. The
score adds up $d_k$ products:

$$q \cdot k = \sum_{t=1}^{d_k} q_t\, k_t .$$

Each product $q_t k_t$ has average 0 and variance $\mathbb E[q_t^2]\,\mathbb E[k_t^2] = 1$.
Variances of independent terms **add**, so the score has variance $d_k$, and its typical
size is $\sqrt{d_k}$. With $d_k = 64$, scores land around $\pm 8$.

Why is that bad? Differentiate softmax's weight $a_j = e^{s_j}/Z$ (with $Z = \sum_k e^{s_k}$)
with respect to its own score, using the quotient rule:

$$\frac{\partial a_j}{\partial s_j} = \frac{e^{s_j} Z - e^{s_j} e^{s_j}}{Z^2} = a_j\,(1 - a_j) .$$

That's large only when $a_j$ is somewhere in the middle. Scores 16 apart make one weight
$e^{16} \approx 9$ million times another: every weight is nearly 1 or nearly 0, so every
$a_j(1 - a_j)$ is nearly 0, and nudging a score changes almost nothing. Learning stalls.

The fix follows directly: divide the score by its typical size, $\sqrt{d_k}$, and the variance
is back to 1, where softmax stays soft. A one-character fix to a real training problem.

Try it: pick a word and see where it looks. Then raise the sharpness, which is what
unscaled dot products do, and watch softmax harden into all-or-nothing.

::figure[attention-heatmap]

**Why it mattered.** The entire mechanism is two matrix multiplications and a softmax,
exactly the operation GPUs are fastest at. Nothing in it is sequential.

**Go deeper:** [Self-attention](/concepts/self-attention) · [Softmax](/concepts/softmax) ·
[Matrix multiplication](/concepts/matrix-multiplication)

## 3.2.2 · Multi-head attention

> **The paper says:** *"Multi-head attention allows the model to jointly attend to
> information from different representation subspaces at different positions. With a
> single attention head, averaging inhibits this."*

**In plain words.** One attention pattern is one *question* per word. But a word usually
needs several answers at once. For "it", the model may want to know **what it refers to**
(the animal), **what's happening to it** (tired), and **where in the sentence it sits**. A
single blend mixes all of that into one average: the blurring section 2 warned about.

Why can't one attention do both? Because one attention gives "it" **one set of weights**,
one way of splitting its attention across the sentence. If one question wants all of it on
"animal" and another wants all of it on "tired", no single split satisfies both. The best
compromise is 50/50, and "it" receives half of each value, blended together. Two questions
need two sets of weights: **two attentions**.

So the paper runs **8 attentions side by side**, called **heads**. Each word vector has
$d_{\text{model}} = 512$ numbers. Rather than one head working on all 512, each head gets its
own learned projection down to $d_k = 64$ numbers and runs its own attention there, with its
own pattern. The 8 answers, each 64 long, are glued back into 512 and mixed by one more
matrix $W^O$:

$$\operatorname{MultiHead}(Q,K,V) = \operatorname{Concat}(\text{head}_1, \ldots, \text{head}_8)\, W^O,
\qquad \text{head}_i = \operatorname{Attention}(QW_i^Q,\, KW_i^K,\, VW_i^V)$$

**The neat part: it's free.** Count the weights. One head working on all 512 numbers needs a
$512 \times 512$ query matrix. Eight heads of 64 need eight $512 \times 64$ matrices: side by
side, that's $512 \times (8 \times 64) = 512 \times 512$. Same for keys and values. Eight
heads of size 64 cost exactly as much as one head of size 512; the same multiplications are
just split into eight groups. (Why shrink each head to $d_{\text{model}}/8$? Precisely so the
cost stays fixed. The price is that each head compares queries and keys with only 64
numbers.)

::video[multi-head-split]

**Why it mattered.** Nobody tells the heads what to look for; they specialise during
training. Later research found heads that track the previous word, heads that link pronouns
to nouns, and "induction heads" that spot a repeated pattern and copy what came next, a
building block of how language models learn from the prompt.

**Go deeper:** [Multi-head attention](/concepts/multi-head-attention)

## 3.2.3 · Three ways the model uses attention

> **The paper says:** *"We need to prevent leftward information flow in the decoder to
> preserve the auto-regressive property."*

**In plain words.** The same attention function appears in three places, with different
rules about who may look at whom:

1. **Encoder self-attention.** English words look at all English words, before and after.
   Understanding a sentence often needs words that come later.
2. **Masked decoder self-attention.** German words look at the German written **so far**,
   never ahead. At translation time the future words don't exist yet, so during training
   the model mustn't be allowed to peek at them. The trick is to set every "look ahead"
   score to $-\infty$ before the softmax, so its weight becomes exactly 0. The score table
   becomes a triangle. That's the "preventing leftward information flow" in the quote:
   in their figures, the future is to the right, so information mustn't flow from it.
3. **Encoder–decoder attention** (today usually called **cross-attention**). The **queries**
   come from the German being written; the **keys and values** come from the encoder's
   English vectors. This is the interpreter glancing back at what they heard: "I'm about to
   write a verb, where's the verb in the English?"

**Why it mattered.** The masked version is the one that took over the world. GPT, Claude
and Llama are a decoder with no encoder: just masked self-attention, trained to predict the
next word of ordinary text. The triangle mask lets one sentence of $T$ words provide $T$
training examples at once, one per position.

**Go deeper:** [Causal masking](/concepts/causal-masking) ·
[Language modeling](/concepts/language-modeling)

## 3.3 and 3.4 · Feed-forward layers, embeddings and the output

> **The paper says:** *"This consists of two linear transformations with a ReLU activation
> in between."*

**In plain words.** After attention lets words **talk**, each word **thinks on its own**.
Every block has a small neural network, applied separately to each word's vector:

$$\operatorname{FFN}(x) = \max(0,\; xW_1 + b_1)\,W_2 + b_2$$

It widens the vector from 512 to 2,048 numbers, applies ReLU (keep positives, zero out
negatives), and narrows it back to 512. It's the ordinary two-layer network from the
neural-networks pages, nothing more. Attention gathers context; the feed-forward network
works on it. Researchers later found that much of what a model "knows" (facts like
"Paris is in France") is stored in these layers.

**Embeddings at both ends.** Words enter as **embeddings**: one learned vector of 512
numbers per token in the vocabulary, looked up in a table. At the other end, the final
vector is turned back into a score for every token in the vocabulary, and softmax makes
those scores a probability for the next word. The paper **shares one table** for both
directions (*"we share the same weight matrix between the two embedding layers and the
pre-softmax linear transformation"*), which saves parameters and makes sense: a token's
vector should look similar going in and coming out.

The paper also multiplies the embeddings by $\sqrt{d_{\text{model}}} \approx 22.6$ when
they go in, without saying why. Here's the usual reading, in numbers. The position stamps
added next (section 3.5) are 256 (sine, cosine) pairs, each pair of length 1, so a stamp has
length $\sqrt{256} = 16$. Embeddings are commonly started with numbers of size about
$1/\sqrt{512}$, which gives a vector of length about 1. Added as they are, the stamp would be
16 times bigger than the word: position would drown out meaning. Times 22.6, the word is the
bigger of the two again.

**Why it mattered.** The "talk, then think" rhythm of attention followed by a feed-forward
network is still the shape of every transformer block today.

**Go deeper:** [Embeddings](/concepts/embeddings) · [Neurons](/concepts/neurons) ·
[Activation functions](/concepts/activation-functions) · [Tokenization](/concepts/tokenization)

## 3.5 · Positional encoding: where is each word?

> **The paper says:** *"Since our model contains no recurrence and no convolution, in order
> for the model to make use of the order of the sequence, we must inject some information
> about the relative or absolute position of the tokens in the sequence."*

**In plain words.** Attention has a blind spot: it **doesn't see order**. It compares every
word with every word, and the result is the same if you shuffle the sentence. "Dog bites
man" and "man bites dog" would look identical. An RNN gets order for free because it reads
in order. A transformer has to be **told**.

The fix is to **add a position stamp** to each word's embedding before the first layer: a
vector that's different for position 0, 1, 2, … What should the stamp be? Let's try the
obvious options.

**Try: the position number itself**, $0, 1, 2, \ldots$ It grows without limit. At position
500, the stamp is hundreds of times bigger than the word's meaning and swamps it.

**Try: the position divided by the length**, so it stays between 0 and 1. Now "the next
word" is a step of $1/10$ in a 10-word sentence and $1/1{,}000$ in a 1,000-word one. The same
distance looks different in every sentence.

**Try: count in binary.** Write the position as bits:

| Position | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| bit 0 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 |
| bit 1 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 1 |
| bit 2 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 |

Every number stays 0 or 1, however long the text, and every position gets a different
pattern. Look at the rows: each bit **repeats** in a cycle, and each cycle is **twice as
long** as the one above. Bit 0 flips every word, bit 1 every 2 words, bit 2 every 4. That's
a row of clocks, each running at half the speed of the last. Binary is almost what we want.
Its only flaw is that it **jumps**: 0, then suddenly 1. Neural networks learn best from
values that change smoothly.

**Make the clocks smooth.** Replace each flipping bit with a smooth wave of the same rhythm:
a sine. One sine isn't quite enough, because it rises and falls and so passes through each
value twice per cycle: $\sin$ alone can't tell "on the way up" from "on the way down". Pair it
with a cosine and the two together are the coordinates of a point going round a circle:
**the angle of a clock hand**, which is unambiguous within a turn.

**Choose the speeds.** Binary halves the speed at each step: the speeds form a geometric
sequence. The paper does the same with a gentler ratio: the first clock turns 1 radian per
word, and the speeds shrink by equal ratios down to about $1/10{,}000$ for the last. With
$d/2$ clocks, that's $\omega_i = 10000^{-2i/d}$ for clock $i = 0, 1, \ldots$, and the stamp
is every clock's (sine, cosine) reading:

$$PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \qquad
PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

The 10,000 is a choice, not a derivation: it sets the slowest clock to take about 60,000
words per turn, far longer than any sentence the paper translated.

Picture **a row of clocks**, each with one hand. To stamp position 37, read off the angle of
every hand. Fast clocks tell nearby positions apart; slow clocks tell far-apart positions
apart. Together, every position gets a unique fingerprint.

::video[positional-clocks]

> *"We chose this function because we hypothesized it would allow the model to easily learn
> to attend by relative positions, since for any fixed offset $k$, $PE_{pos+k}$ can be
> represented as a linear function of $PE_{pos}$."*

We can check that claim. Take one clock with speed $\omega$ and use the angle-addition
formulas on its reading $k$ words later:

$$
\begin{aligned}
\sin(\omega(pos + k)) &= \cos(\omega k)\,\sin(\omega\, pos) + \sin(\omega k)\,\cos(\omega\, pos), \\
\cos(\omega(pos + k)) &= -\sin(\omega k)\,\sin(\omega\, pos) + \cos(\omega k)\,\cos(\omega\, pos).
\end{aligned}
$$

As a matrix:

$$\begin{pmatrix} \sin(\omega(pos+k)) \\ \cos(\omega(pos+k)) \end{pmatrix}
= \begin{pmatrix} \cos\omega k & \sin\omega k \\ -\sin\omega k & \cos\omega k \end{pmatrix}
\begin{pmatrix} \sin(\omega\, pos) \\ \cos(\omega\, pos) \end{pmatrix} .$$

The matrix contains $k$ but **not** $pos$. So "$k$ words later" is one fixed rotation of
every clock, wherever you start: a linear function, exactly as claimed. "Look three words
back" is something a network can learn once and reuse everywhere. (Rotary embeddings, RoPE,
take this one step further and apply the rotation to queries and keys directly.) Move the
position below and watch each clock's hand turn:

::figure[positional-encoding-explorer]

**Why it mattered.** The paper also tried *learned* position vectors and found *"nearly
identical results"*. Many later models (GPT-2, BERT) learned them; most new ones use a
descendant of this idea, rotary embeddings (RoPE), which rotate queries and keys instead of
adding a stamp.

**Go deeper:** [Positional encoding](/concepts/positional-encoding)

## 4 · Why self-attention? Counting the cost

> **The paper says:** *"…the total computational complexity per layer, the amount of
> computation that can be parallelized, … and the path length between long-range
> dependencies in the network."*

**In plain words.** Section 4 compares the three options on three counts. For $n$ words
with vectors of size $d$ (Table 1 in the paper):

| Layer type | Work per layer | Steps that must wait for each other | Longest path between two words |
|---|---|---|---|
| Self-attention | $n^2 \cdot d$ | 1 | 1 |
| Recurrent | $n \cdot d^2$ | $n$ | $n$ |
| Convolutional (window $k$) | $k \cdot n \cdot d^2$ | 1 | $\log_k n$ |

Each entry is a count you can do yourself:

- **Attention:** $n \times n$ scores, each a dot product of length $d$: $n^2 d$.
- **Recurrent:** at each of $n$ steps, the memory vector (size $d$) is multiplied by a
  $d \times d$ matrix: $n d^2$.
- **Convolution:** at each of $n$ positions, $k$ neighbours each go through a $d \times d$
  matrix: $k n d^2$. Each layer widens what a word can see by a factor of about $k$, so
  covering $n$ words takes about $\log_k n$ layers (dilated convolutions, as the paper notes).

So attention is cheaper than recurrence exactly when $n^2 d < n d^2$, that is, when
$n < d$: when the sentence is shorter than the vector.

- **Steps that must wait:** an RNN must finish word 1 before word 2. Attention does all
  words in one go.
- **Longest path:** attention connects any two words directly.
- **Work:** here attention pays. Every word looks at every word, so the cost grows with
  $n^2$. For sentences ($n \approx 30$, $d = 512$) that's *cheaper* than an RNN. For a
  whole book it gets expensive.

**Why it mattered.** That $n^2$ is the reason chatbots have a **context window** limit,
and why so much later research went into cheaper attention. The paper saw it coming and
suggested restricting attention to a nearby window for very long inputs.

**Go deeper:** [Self-attention](/concepts/self-attention)

## 5 · Training: warm up, then cool down

> **The paper says:** *"This corresponds to increasing the learning rate linearly for the
> first warmup_steps training steps, and decreasing it thereafter proportionally to the
> inverse square root of the step number."*

**In plain words.** The model was trained on about **4.5 million** English–German sentence
pairs (and 36 million English–French), with the **Adam** optimizer, on **8 GPUs**: 12 hours
for the base model, 3.5 days for the big one.

The detail people still copy is the **learning-rate schedule**:

$$\text{lrate} = d_{\text{model}}^{-0.5} \cdot \min\!\big(\text{step}^{-0.5},\;
\text{step} \cdot \text{warmup\_steps}^{-1.5}\big), \qquad \text{warmup\_steps} = 4000$$

It looks scary; it's a ramp and a slope. For the first 4,000 steps the learning rate
**climbs in a straight line** (the second term is smaller). After that it **slowly decays**
like $1/\sqrt{\text{step}}$ (the first term is smaller). The two terms are equal where
$\text{step}^{-0.5} = \text{step} \cdot 4000^{-1.5}$, that is, at step 4,000, so that's the
peak: $512^{-0.5} \cdot 4000^{-0.5} \approx 0.0007$. The shape is an engineering choice the
authors found to work, not something derived. Why warm up? At the start the
weights are random and Adam's running averages of the gradients are still unreliable, so big
steps can throw the model somewhere it never recovers from. Start gently, speed up once
things settle, then slow down to fine-tune.

Two regularisers keep the model from memorising its training data:

- **Dropout** ($P = 0.1$): randomly switch off 10% of the values during training, so the
  model can't rely on any single one.
- **Label smoothing** ($\epsilon = 0.1$): train it to put 90% probability on the correct
  word rather than 100%, spreading the rest. *"This hurts perplexity, as the model learns
  to be more unsure, but improves accuracy and BLEU score."* A slightly humble model
  translates better.

**Why it mattered.** Warmup followed by decay is still the default recipe for training
transformers, and large models usually fail to train without the warmup.

**Go deeper:** [Adam](/concepts/adam-optimizer) · [Learning rate](/concepts/learning-rate) ·
[Log-loss](/concepts/log-loss)

## 6 · Results: better, faster, cheaper

> **The paper says:** *"On the WMT 2014 English-to-German translation task, the big
> transformer model … outperforms the best previously reported models (including
> ensembles) by more than 2.0 BLEU."*

**In plain words.** **BLEU** scores a translation by how many of its word sequences also
appear in human reference translations; higher is better, and gains of 1 point are
considered solid. The big transformer scored **28.4** on English→German, beating even
*ensembles* (several models voting together) by over 2 points, and **41.8** on
English→French, a new single-model record. It did so for a **fraction of the training
cost** of the systems it beat.

The paper also takes its own model apart (Table 3) to see which pieces matter:

- **Heads:** one head is clearly worse than eight, but too many heads hurts again.
- **Smaller $d_k$** (shorter keys) hurts: matching queries to keys needs room.
- **Bigger is better,** and dropout matters a lot.
- **Sine-wave or learned positions:** about the same.

Finally, they test it on a different task, English **parsing** (finding the grammatical
structure of a sentence), with little tuning, and it does well there too: a hint that this
wasn't just a translation trick.

**Why it mattered.** Better quality *and* faster training is rare. Faster training meant
more data could be used, and the rest of the story is what happened when people used a
lot more.

## 7 · Conclusion, and what happened next

> **The paper says:** *"We are excited about the future of attention-based models and plan
> to apply them to other tasks."*

That turned out to be an understatement. Within two years, the two halves of the
transformer went their separate ways:

- **Decoder only → GPT** (2018 onwards). Keep the masked decoder, drop the encoder and
  cross-attention, and train it to **predict the next word** of huge amounts of text. Scale
  it up and you get GPT-3, ChatGPT, Claude, Llama.
- **Encoder only → BERT** (2018). Keep the encoder, train it to fill in hidden words, and
  you get a model that *understands* text, which powered search and classification.

Today's models have small refinements (normalising *before* each sub-layer instead of
after, rotary positions, other activation functions, faster attention), but you'd
recognise the block from this paper inside every one of them.

**Build it yourself.** This site's [transformer track](/tracks/transformers) builds the
decoder-only version from a NumPy array upward: every piece above, as maths and runnable
code, ending with a tiny GPT you train in about 20 seconds.
