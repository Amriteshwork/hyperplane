---
title: BERT
summary: The transformer's encoder, reading the whole input in both directions. Pretrain it once by filling in blanks, then fine-tune it for classification, tagging, search or question answering.
area: sequence-models
prerequisites: [transformer, masked-language-modeling]
---

You've built a GPT. It reads **left to right** and is trained to **continue** text, which
makes it a writer. In 2018, a team at Google asked what you'd get if you kept the same
blocks but built a **reader** instead: a model whose only job is to understand the text in
front of it, all of it at once, so that other tasks can be built on top. They called it
**BERT**: **B**idirectional **E**ncoder **R**epresentations from **T**ransformers.

**The whole difference is one mask and one objective.**

| | GPT (decoder) | BERT (encoder) |
|---|---|---|
| each token attends to | the tokens before it | **every** token |
| attention mask | causal triangle | padding only |
| pretraining task | predict the next token | fill in the **masked** tokens |
| output | next-token scores | **one vector per token** |
| good at | generating text | classifying, tagging, searching, extracting |

Here's what removing the mask means. Turn the causal mask on, then off: off is BERT.

::figure[attention-heatmap]

With the mask off, "the" can see the noun it introduces, and "it" can see "slept" as well
as "animal". Each token's vector now describes it **in the light of the whole sentence**.
"bank" in "river bank" and "bank" in "bank account" come out as different vectors. Those
**contextual embeddings** are BERT's product.

### What goes in: [CLS] A [SEP] B [SEP]

Many tasks are about **two** pieces of text: a question and a passage, or a premise and a
claim. BERT packs both into one input:

- **`[CLS]`** always comes first. It belongs to no word, so its final vector is free to
  summarise the whole input. A classifier reads it.
- **`[SEP]`** closes each sentence.
- Every token's input vector is the **sum of three embeddings**: *which token*, *which
  sentence* (A or B, the **segment** embedding), and *which position*.

$$x_t = E_{\text{token}}[\,\text{id}_t\,] + E_{\text{segment}}[\,A \text{ or } B\,] + E_{\text{position}}[\,t\,]$$

Type your own two sentences and see the three rows add up:

::figure[bert-input]

### Pretraining: learn language from unlabelled text

BERT was pretrained on 3.3 billion words of Wikipedia and books with two tasks:

1. **Masked language modeling**: hide 15% of the tokens (80% `[MASK]`, 10% random, 10%
   unchanged) and predict them from both sides. This is where the bidirectionality comes
   from.
2. **Next sentence prediction (NSP)**: half the time B really follows A, half the time
   it's a random sentence. Predict which, from `[CLS]`. The idea was to teach relations
   between sentences. Later work (RoBERTa, 2019) found that dropping NSP did no harm,
   and most of BERT's successors skip it.

::figure[fill-in-the-blank]

### Fine-tuning: one model, many tasks

This is BERT's real contribution. Pretraining is expensive (four days on 16 TPU chips for BERT-base) and
it's done **once**. Every task after that starts from the pretrained weights, adds a tiny
**head**, and trains everything briefly on labelled data. The paper says every result it
reports can be reproduced in **at most an hour** on one Cloud TPU, or a few hours on a GPU.

- **Classify a sentence or a pair** (sentiment, entailment, "are these duplicates?"):
  $$p = \operatorname{softmax}(C\,W^\top), \qquad C = \text{the final } [\text{CLS}] \text{ vector},\ W \in \mathbb{R}^{K\times H}$$
- **Tag every token** (names, places, dates): a small classifier on **each** token's
  vector.
- **Answer a question from a passage**: pack question as A and passage as B, and learn
  two vectors $S$ and $E$. The answer starts at the token $i$ that maximises $S\cdot T_i$
  and ends at the token $j \ge i$ that maximises $E\cdot T_j$.

Pick a task and see which output vectors it reads:

::figure[bert-heads]

In the paper, this simple recipe set new records on **eleven** benchmarks at once,
including GLUE (80.5, up 7.7 points) and SQuAD question answering. A head trained for an
hour beat systems that had been carefully engineered for a single task.

### The sizes

**BERT-base**: 12 layers, $d = 768$, 12 heads, 110M parameters. That's the same shape as
the original GPT, chosen so the paper could compare reading both ways against reading left to
right, fairly. **BERT-large**: 24 layers, $d = 1024$, 16 heads, 340M. The block is the
same one you built on the Transformer page. The original used learned position
embeddings, post-LN and GELU. Our code keeps the sinusoidal positions and pre-LN from
earlier pages, which doesn't change the idea.

### What the code shows

The **NumPy** version imports GPT's `block` and `init_params` from the Transformer page
and hands the block a different mask, one that hides only padding. It then checks the
three things that make an encoder an encoder:

- change the **last** word, and the **first** word's vector changes (GPT: never);
- add `[PAD]` tokens, and nothing changes for the real ones;
- an untrained MLM head scores $\log V$ on the masked words; a fine-tuning head is one
  $d \times K$ matrix on the `[CLS]` vector.

The **PyTorch** version builds TinyBERT from `Block(d, n_heads, causal=False)` and
pretrains it with masking for about 20 seconds on the generated sentences: *"zoe likes
jam, so zoe eats jam."*. Then it hides the **first** word: *"[MASK] likes figs, so raj
eats figs."* At that position a GPT has seen nothing, so it could only guess one name in
eight. TinyBERT answers **raj** every time, because it reads the copy on the right. Hide
both copies, and it's back to guessing.

### Where BERT is today

Generation went to GPT-style decoders. But encoders never went away: search engines rank
results with them, retrieval systems (RAG) embed documents with them, and classifiers in
production run on them because they're small and fast. RoBERTa, DistilBERT, DeBERTa and
ModernBERT are all this page's model, with better training.
