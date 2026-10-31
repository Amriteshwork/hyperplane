---
title: Chat fine-tuning
summary: A pretrained model can't answer questions. It only continues text. Fine-tuning on example conversations, with a loss that only counts the assistant's words, turns a text-continuer into something that replies, and stops.
area: sequence-models
prerequisites: [llm-training, text-generation]
videos:
  chat-is-one-sequence:
    title: A conversation is just one long sequence
    youtube: SRLeIyEKq7o
---

Train a GPT on a slice of the internet and type *"what's the capital of France?"*. You might
get *"what's the capital of Germany? what's the capital of Spain?"*. It isn't being difficult.
It's doing what it was trained to do: **continue the text**, and a question in a web page is often
followed by more questions.

That's a **base model**. It knows a lot, but it isn't an assistant. Turning it into one takes no
new architecture, only more training, on the right data.

### A conversation is one long sequence

There's no separate "user input" channel in a GPT. A chat is flattened into one string of tokens,
with special markers to say who's talking. Real models each define their own format. OpenAI's
ChatML looks like this:

```text
<|im_start|>user
What does zoe eat?<|im_end|>
<|im_start|>assistant
jam.<|im_end|>
```

The model still predicts the next token. It's learnt that after `<|im_start|>assistant`, the next
tokens are a helpful reply, and that `<|im_end|>` comes when the reply is finished. **Knowing when
to stop is itself something it has to learn.** Without that it would write the user's next message
too.

::video[chat-is-one-sequence]

### Teach only the assistant's turns

The model mustn't be trained to predict what the user will say next. So only the assistant's tokens
count toward the loss. The others get a special label, $-100$, that PyTorch's cross-entropy ignores:

::figure[chat-labels]

The stop marker counts as one of the assistant's tokens, and that's how the model learns to stop.

The rest is ordinary. It's the same loss, the same optimizer, and the same weights the base model
already has, usually with a lower learning rate, so the new data adjusts what it already knows
without overwriting it. This step is called **supervised fine-tuning**: *supervised* since each
example comes with a right answer, written by a person.

### See the change

We tried this for real on the tiny GPT. Pretrain on plain text that has no questions in it. Then
fine-tune on just **16 example conversations**, each with a fact, a question, and a one-word answer
that has to come from the fact:

::figure[base-vs-chat]

Three things to notice:

1. **The base model scores 0%.** It knows the words, but nothing has told it this is a question.
2. **Fine-tuning takes it to about 90%** on conversations it never saw, stop token included, in only
   80 steps.
3. **Pretraining helps.** The same 16 chats and steps with no pretraining reached 58%. That gain
   is the reason for the two stages: raw text is plentiful and cheap, while conversations written by
   people are scarce and expensive.

We should be honest about the size of that last result. With 200 conversations, the model with no
pretraining caught up completely. The pretraining advantage is about **having little labelled data**,
and that's the situation real companies are in, relative to the size of their models.

### What this doesn't give you

Fine-tuning copies the style of the examples. It has no sense of which of two possible replies people
would *prefer*, and writing a perfect reply to every kind of question is impossibly expensive. Asking
people to compare two replies is far cheaper. That's the next step: [RLHF](/concepts/rlhf).

**What the code shows.** The NumPy file builds the template and the labels, and proves the loss can't
be changed by anything predicted during the user's turn. The PyTorch file runs the experiment above
in about 30 seconds.
