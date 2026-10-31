---
title: Encoder-decoder transformer
summary: Two transformers working as a pair. The encoder reads the whole input at once; the decoder writes the output one token at a time, reading the encoder through cross-attention. The original transformer, built for translation.
area: sequence-models
prerequisites: [transformer, cross-attention]
---

You've built a GPT, a **decoder**: it writes text one token at a time, each token seeing
only the ones before it. If you've read the BERT page, you've also met an **encoder**: the
same blocks with the mask taken off, so every token reads the whole input. Each one is
good at half a job. A GPT writes but has nothing to translate *from*. BERT understands
its input but can't write anything.

**Translation needs both halves.** First read the English sentence, all of it, both
directions. Then write the German, one word at a time, looking back at the English
whenever you need to. That's the **encoder-decoder** transformer, the one introduced in
*Attention Is All You Need*, and it's built entirely from pieces you already have:

| | encoder | decoder |
|---|---|---|
| reads | the source (English) | the target written so far (German) |
| self-attention mask | none: every word sees every word | causal: only earlier words |
| extra layer | | **cross-attention** into the encoder's output |
| runs | **once** per sentence | once per output word |
| produces | one vector per source word (the **memory**) | next-word scores |

The decoder block has three steps instead of two: look back at the German so far (masked
self-attention), **read the English** (cross-attention), then think (the MLP). Each step
is a residual with its own layer norm.

$$h \leftarrow h + \text{SelfAttn}_{\text{causal}}(\text{LN}(h)), \qquad
h \leftarrow h + \text{CrossAttn}(\text{LN}(h),\ \text{memory}), \qquad
h \leftarrow h + \text{MLP}(\text{LN}(h))$$

Step through a translation and watch the three kinds of attention take turns:

::figure[seq2seq-steps]

### Training: teacher forcing

Training looks just like GPT's. The decoder is shown the **true** German, shifted one place
right with a start token `[BOS]` in front, and asked to predict every next word at once:

| decoder input | `[BOS]` | zoe | mag | marmelade | … | feigen |
|---|---|---|---|---|---|---|
| target | zoe | mag | marmelade | , | … | `[EOS]` |

The causal mask means position $t$ sees only the German up to $t$, while
cross-attention lets it see **all** the English. One cross-entropy over all positions
trains everything together: the encoder, the decoder and the alignment between them. The
end token `[EOS]` is how the model learns to stop. Using the true previous words instead
of the model's own guesses is called **teacher forcing**.

At test time there's no teacher. Encode the English **once**, then start from `[BOS]`,
pick the likeliest next word, append it and repeat until `[EOS]`. That's **greedy
decoding**. Real translators use [beam search](/concepts/text-generation), which keeps the
few best partial translations at every step instead of only one.

### What the model learns to look at

Nobody tells the model which English word a German word comes from. It finds out,
because that's the only way to translate well. Plot the cross-attention weights and the
**alignment** appears: mostly diagonal, with crossings where the two languages order
words differently.

::figure[cross-attention]

### Three families from one design

Take the full encoder-decoder and remove half:

- **Keep both** for sequence-to-sequence tasks, where an input becomes a different output:
  translation, summarisation, speech to text (Whisper), and T5, which casts *every* task
  as text in, text out.
- **Keep only the decoder** and train it to continue text: GPT, Claude, Llama. A chat
  model treats your prompt and its answer as **one** sequence, and causal self-attention
  does the reading that cross-attention did.
- **Keep only the encoder** and train it to fill in blanks: BERT, for classification and
  search.

Decoder-only won for general-purpose chat models, mostly because it's simpler and scales
well on plain text. Encoder-decoders are still common where the input and output are
clearly different things, like audio and its transcript, or a document and its summary.

**What the code shows.** The NumPy version builds the whole forward pass from earlier
pages: the encoder is the transformer's `block` with no mask, and `decoder_block` slots
`cross_attention` between masked self-attention and the MLP. It checks the two rules that
make it work. Every German position sees all the English, but none sees later German.
The PyTorch version trains a 61,000-parameter model for 150 steps (about 15 seconds on a
laptop CPU) to translate *"zoe likes jam, so raj eats figs."* into *"zoe mag marmelade,
also isst raj feigen."*, where the verb jumps ahead of its subject. It gets 100% of
sentences it never saw exactly right, and its cross-attention puts each name and food on
its own English word 97% of the time.
