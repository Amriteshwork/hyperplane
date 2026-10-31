---
title: Sequence Models
summary: Models that read and write text one token at a time, built from recurrence and attention. The ideas behind GPT, BERT and every modern LLM.
order: 8
tracks: [transformers, gpt-to-chat, bert, encoder-decoder, recurrent-networks]
---

Sequence models treat their input as an ordered stream of tokens and learn how each one
depends on the others. They're taught here as guided tracks, from the maths up, because
their parts (tokenization, embeddings, attention, positional encoding) are each a concept of
their own. The transformer comes first, and the next track takes it from a tiny GPT to a chat
model. BERT-style encoders, and the encoder-decoder pairs used for translation, build on the
transformer. Recurrent networks, which it replaced, show why attention was needed: their track
ends by measuring that wall and breaking through it.
