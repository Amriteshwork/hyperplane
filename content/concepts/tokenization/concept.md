---
title: Tokenization
summary: A network can't read letters, only numbers. Tokenization chops text into pieces and gives each piece an id, losslessly.
area: sequence-models
prerequisites: [numpy-basics]
---

Every model on this site so far has eaten a table of numbers. Text isn't one. Before a
transformer can read "the cat sat", someone has to answer a surprisingly deep question:
**what are the units of text, and what number does each one get?**

The answer has three steps:

1. **Split** the text into pieces called **tokens**.
2. **Number** every distinct token. The list of them is the **vocabulary**, and its size is $V$.
3. **Replace** each token with its number, its **token id**.

$$\texttt{"the cat"} \;\xrightarrow{\text{split}}\; [\texttt{t},\texttt{h},\texttt{e},\texttt{␣},\texttt{c},\texttt{a},\texttt{t}] \;\xrightarrow{\text{look up}}\; [5, 4, 3, 0, 2, 1, 5]$$

Decoding runs the same table backwards. A good tokenizer is **lossless**:
$\text{decode}(\text{encode}(s)) = s$ for every string, so nothing the writer typed is ever
thrown away.

**How big should a token be?** It's a trade-off, and each extreme fails in its own way:

| Tokens are… | Vocabulary $V$ | Sequence length | The catch |
|---|---|---|---|
| characters | tiny (~100) | very long | the model must learn spelling, and long sequences cost compute (attention grows as length²) |
| whole words | huge (100k+) | short | "tokenizes", "tokenizing", "tokenizer" are unrelated ids; a new word is simply unknown |
| **subwords** | medium (~50k) | medium | the sweet spot every modern LLM uses |

**Byte-pair encoding (BPE)** finds the subwords automatically, and it's beautifully simple.
Start from single characters (really, bytes), then repeat: *find the pair of neighbouring
tokens that appears most often, and glue it into one new token.* On English, "t"+"h" merges
early, then "th"+"e", then " the". Common words end up as single tokens and rare words split
into familiar pieces ("token" + "ization"), so nothing is ever unknown. The code below runs
a few merges by hand, and you can watch the sequence shrink.

**Why this matters later.** Tokenization explains many famous LLM quirks. Models are bad at
counting the r's in "strawberry" because they never see letters, only chunks. Arithmetic
is odd because "1234" might be one token and "1235" two. And a model's "context length"
is counted in tokens, not words.

The ids themselves mean nothing: id 6 isn't "bigger" than id 3. Turning ids into something
a network can do maths with is the job of the next concept, **embeddings**.
