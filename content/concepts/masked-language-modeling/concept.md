---
title: Masked language modeling
summary: Hide some of the words and train the model to fill them in, using the text on both sides of each gap. The pretraining task behind BERT.
area: sequence-models
prerequisites: [language-modeling, causal-masking]
---

You already know one way to learn from raw text: **predict the next token**. It needs no
labels, since the text is its own answer key. But it has a blind spot built in. At every
position the model sees only the **left**: to guess the word after "the", it gets "the" and
nothing more. That's the right rule for *writing* text, where the future doesn't exist yet.
It's the wrong rule for *understanding* text, where the whole sentence is already on the page.

Try reading this the way a GPT has to:

> *The ____ barked at the postman all morning.*

Left to right, at the gap you've seen only "The". It could be anything. Read the whole
sentence and the answer is obvious: **dog**. The clue was on the right.

**Masked language modeling (MLM) is a fill-in-the-blanks test.** Take a sentence, hide a
few of its words behind a special `[MASK]` token, and train the model to recover them
from **everything else**: the left *and* the right. Teachers call this a *cloze test*, and
it goes back to 1953. The model is scored only on the hidden words:

$$\mathcal{L} = -\frac{1}{|M|}\sum_{t \in M} \log p_\theta\!\left(x_t \;\middle|\; \tilde{x}\right)$$

$M$ is the set of hidden positions, $x_t$ the word that was really there, and
$\tilde{x}$ the sentence with its holes. It's the same cross-entropy as next-token
prediction. Only the question has changed.

Click words to hide them, then switch between the two kinds of reader:

::figure[fill-in-the-blank]

**Why not just remove the causal mask from a GPT?** Because then the next-token task
becomes cheating: every position could look one step ahead and copy its answer. Masking
fixes this from the other side. Attention can now go everywhere, so the answer itself
has to be **taken out of the input**. A model can't copy a word that isn't there.

**BERT's recipe: 15%, then 80 / 10 / 10.** BERT picks 15% of the tokens in each
sequence. Of those:

- **80%** are replaced by `[MASK]`;
- **10%** are replaced by a **random** token;
- **10%** are **left unchanged**.

All of them are predicted, all of them are scored. Why the strange split? `[MASK]` never
appears in real text, and after pretraining the model only ever sees real text. A model
that has only ever had to fix `[MASK]`s could learn "an ordinary word is always correct,
ignore it". With some words secretly swapped and some secretly kept, it can't trust any
visible word blindly. It has to build a good representation for **every** position.

**Why 15%?** It's a trade-off. Hide too few, and each sentence teaches very little, so
training is slow. Hide too many, and too little context is left to work out the answers
from. BERT's authors chose 15%. Later work found that bigger models can cope with more.

**The price.** A GPT gets a lesson at *every* position of every sentence. An MLM gets one
at only 15% of them. MLM learns richer, two-sided representations, but it needs more
passes over the data to see as many examples.

**What the code shows.** The NumPy version needs no neural network at all. It takes the
bigram table from Language modeling and uses it in both directions: what usually comes
after the left neighbour, *and* what usually comes before the right one. Guessing hidden
characters that way cuts the loss from 1.31 to 0.49 nats. The PyTorch version trains
two tiny models with the real 80/10/10 recipe, one reading the left neighbour and one
reading both. Reading both nearly halves the loss. That two-neighbour model is word2vec's
CBOW, from 2013. Swap its two neighbours for "every token, weighted by attention" and you
have BERT.
