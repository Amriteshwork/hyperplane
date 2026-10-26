---
title: Sequence Models
summary: Models that read and write text one token at a time, built from recurrence and attention. The ideas behind GPT, BERT and every modern LLM.
order: 8
tracks: [transformers, recurrent-networks]
---

Sequence models treat their input as an ordered stream of tokens and learn how each one
depends on the others. They're taught here as guided tracks, from the maths up, because
their parts (tokenization, embeddings, attention, positional encoding) are each a concept of
their own. The transformer comes first; recurrent networks, which it replaced, show why
attention was needed. BERT-style encoders and their descendants build on the transformer.
