---
title: BERT, explained
summary: The 2018 paper that taught a transformer to read in both directions, by filling in blanks, and made "pretrain once, fine-tune for everything" the standard recipe. Section by section, in plain words, with the key ideas animated.
order: 2
source: devlin-bert
concepts: [embeddings, causal-masking, multi-head-attention, layer-normalization, language-modeling, transformer, masked-language-modeling, bert]
track: bert
videos:
  two-readers:
    title: "Two readers: left to right, or the whole sentence at once"
    youtube: UN0n47EsfLg
  masking-recipe:
    title: "Hide 15%, then 80 / 10 / 10"
    youtube: BvQKYtMbpEs
  input-sum:
    title: "Token + segment + position: BERT's input, added up"
    youtube: ZN21iYr34n4
  next-sentence:
    title: "Next sentence prediction: does B follow A?"
    youtube: MjM8xt5xOaE
  pretrain-finetune:
    title: "Pretrain once, fine-tune many times"
    youtube: kxcq68_IneI
---

In October 2018, four researchers at Google AI Language posted a paper with an unusual
results section. It didn't beat the best system on one benchmark. It beat it on
**eleven**, from sentiment to question answering, often by large margins, and every time
with the **same pretrained model** and a tiny task-specific layer on top.

The model was **BERT**. Within a year it was ranking Google Search results, and "BERT-like"
had become the default way to build a text classifier. The paper's method of
**pretraining a big model on raw text and then fine-tuning it** on each task is now how
almost every language model is made, ChatGPT included.

**How to read this page.** Each section below follows one section of the paper, in order.
Each one starts with **what the paper says** (a short quote), then explains it **in plain
words**, shows it in a figure or animation, and says **why it mattered**. Links at the end
of each section go to this site's concept pages, which build each idea from scratch with
maths and runnable code. If you've read the *Attention Is All You Need* walkthrough, you
already know the blocks BERT is made of. This paper is about what to **train** them on.

## Abstract · A transformer that reads both ways

> **The paper says:** *"BERT is designed to pre-train deep bidirectional representations
> from unlabeled text by jointly conditioning on both left and right context in all
> layers."*

**In plain words.** Unpack that sentence one phrase at a time:

- **Pre-train … from unlabeled text.** Learn from ordinary text (Wikipedia, books) with
  no human labels. The text supplies its own questions and answers.
- **Representations.** BERT's output isn't a prediction. It's **one vector per token**,
  a description of that token in context, for other tasks to build on.
- **Both left and right context.** Each token's vector is computed from the words before
  it **and** after it.
- **In all layers.** Not two half-models glued together at the end, but both directions
  mixed together in every one of the 12 or 24 layers.

The second half of the abstract is the payoff: add **one output layer**, fine-tune, and
you get state-of-the-art results *"without substantial task-specific architecture
modifications"*.

**Why it mattered.** Before BERT, each NLP task had its own carefully designed
architecture. After it, most tasks became "BERT plus one layer".

**Go deeper:** [BERT](/concepts/bert) · [Embeddings](/concepts/embeddings)

## 1 · Introduction: the trouble with reading left to right

> **The paper says:** *"The major limitation is that standard language models are
> unidirectional, and this limits the choice of architectures that can be used during
> pre-training."*

**In plain words.** The best pretrained model at the time was OpenAI's **GPT**, a
transformer trained to predict the next word. To make that task fair, GPT uses a
**causal mask**: every token can only attend to tokens on its left. Otherwise it would
just read the answer one position ahead.

That's fine for writing. It's a handicap for **understanding**. Consider:

> *The ____ barked at the postman all morning.*

At the gap, a left-to-right model has seen only "The". The word that gives it away,
"barked", is on the right, out of sight. The paper argues this hurts most on tasks like
question answering, where *"it is crucial to incorporate context from both directions"*.

::video[two-readers]

So why not just remove the mask? Then each word could "see itself" through the layers
above it, and predicting it would be trivial. That's the puzzle the paper solves. Its
answer, the **masked language model**, is in section 3.1.

**Why it mattered.** The paper names the exact assumption everyone had inherited from
language modelling, left to right, and shows it isn't needed for understanding.

**Go deeper:** [Causal masking](/concepts/causal-masking) ·
[Language modeling](/concepts/language-modeling)

## 2 · Related work: features or fine-tuning

> **The paper says:** *"There are two existing strategies for applying pre-trained
> language representations to down-stream tasks: feature-based and fine-tuning."*

**In plain words.** By 2018 everyone agreed that pretraining on raw text helps. They
disagreed on how to *use* the pretrained model:

- **Feature-based (ELMo, 2018).** Run the pretrained model, take its vectors, and feed
  them as extra inputs to your **own** task-specific model. ELMo trained one LSTM
  left-to-right and a second one right-to-left, and **concatenated** their vectors: both
  directions, but never mixed inside the model.
- **Fine-tuning (OpenAI GPT, 2018).** Keep the pretrained model *as* the task model. Add
  one small layer, and keep training **all** the weights on the labelled data.

BERT takes the **fine-tuning** route of GPT, and gets **deep** bidirectionality, the
thing ELMo only approximated by gluing two one-way models together.

**Why it mattered.** Fine-tuning won. Today "take a pretrained model and fine-tune it" is
the default for almost every task, in language and beyond.

**Go deeper:** [Embeddings](/concepts/embeddings) ·
[LSTM](/concepts/lstm) · [Transformer](/concepts/transformer)

## 3 · BERT: one architecture, two steps

> **The paper says:** *"There are two steps in our framework: pre-training and
> fine-tuning."*

**In plain words.** **Pretraining**: train on lots of unlabelled text with made-up tasks.
**Fine-tuning**: start every task from those same weights and train briefly on its
labelled data. The architecture is the same in both, apart from the output layer.

The architecture itself is the **encoder** of the original transformer: blocks of
multi-head self-attention and an MLP, each wrapped in a residual connection and layer
norm, with **no causal mask**. The paper reports two sizes:

| | layers $L$ | width $H$ | heads $A$ | parameters |
|---|---|---|---|---|
| BERT-base | 12 | 768 | 12 | 110M |
| BERT-large | 24 | 1024 | 16 | 340M |

BERT-base was deliberately given the **same size as OpenAI GPT**. *"Critically, however,
the BERT Transformer uses bidirectional self-attention, while the GPT Transformer uses
constrained self-attention."* Same blocks, same size, one mask apart. Turn the mask off
below and you have BERT's attention:

::figure[attention-heatmap]

**Why it mattered.** Because the architecture didn't change, every later improvement to
transformers (bigger, faster, better trained) carried straight over to encoders.

**Go deeper:** [Multi-head attention](/concepts/multi-head-attention) ·
[Layer normalization](/concepts/layer-normalization) · [Transformer](/concepts/transformer)

## 3 · Input: [CLS], [SEP] and three embeddings

> **The paper says:** *"For a given token, its input representation is constructed by
> summing the corresponding token, segment, and position embeddings."*

**In plain words.** BERT must handle one sentence (*"is this review positive?"*) and two
(*"does this passage answer this question?"*) with the same model. So every input is
packed into one sequence:

$$\texttt{[CLS]}\ \ \text{sentence A}\ \ \texttt{[SEP]}\ \ \text{sentence B}\ \ \texttt{[SEP]}$$

- **Tokens** come from **WordPiece**, a 30,000-entry subword vocabulary: "playing"
  becomes "play" + "##ing".
- **`[CLS]`** ("classification") is always first. Its final vector, called $C$, is what
  a classifier reads.
- **`[SEP]`** marks the end of each sentence.
- A **segment embedding**, one learned vector for A and another for B, is added to every
  token, so attention can tell which sentence a word came from.
- A **position embedding** is added too, since attention alone can't see word order.
  BERT learns one vector per position, up to 512.

Every token's input is the **sum** of the three. Type two sentences and watch the rows:

::figure[bert-input]

::video[input-sum]

**Why it mattered.** A single input format for one or two texts meant that one
pretrained model could serve classification, entailment, similarity *and* question
answering, with no changes inside.

**Go deeper:** [Tokenization](/concepts/tokenization) · [Embeddings](/concepts/embeddings) ·
[Positional encoding](/concepts/positional-encoding)

## 3.1 · Task 1: the masked language model

> **The paper says:** *"In order to train a deep bidirectional representation, we simply
> mask some percentage of the input tokens at random, and then predict those masked
> tokens."*

**In plain words.** Here's the trick that makes reading both ways possible. If the model
may look everywhere, take the answer **out of the input**. Replace some words with a
`[MASK]` token and train the model to recover them from both sides. It's a
fill-in-the-blanks test, which the paper calls by its old name, a **Cloze task**
(Taylor, 1953).

The details matter, and the paper spells them out:

- **15%** of the tokens in each sequence are picked for prediction.
- Of those, **80%** become `[MASK]`, **10%** become a **random** token, and **10%** stay
  **unchanged**.
- The loss is cross-entropy on the picked positions only.

Why not always use `[MASK]`? Because `[MASK]` never appears when the model is fine-tuned
on real text. The paper calls this *"a mismatch between pre-training and fine-tuning"*.
With some words secretly swapped and some secretly kept, the model can't tell which
visible words are genuine, so it has to build a good vector for every one of them.

::video[masking-recipe]

Try it: hide a word, then switch between a left-to-right reader and BERT.

::figure[fill-in-the-blank]

**Why it mattered.** Masking solved the puzzle from section 1: a model that sees
everything, trained on a task it can't cheat at. It's the idea the whole paper rests on.

**Go deeper:** [Masked language modeling](/concepts/masked-language-modeling) ·
[Language modeling](/concepts/language-modeling)

## 3.1 · Task 2: next sentence prediction

> **The paper says:** *"Specifically, when choosing the sentences A and B for each
> pre-training example, 50% of the time B is the actual next sentence that follows A
> (labeled as IsNext), and 50% of the time it is a random sentence from the corpus
> (labeled as NotNext)."*

**In plain words.** Masked words teach a model about words in context. Many tasks are
about how two **sentences** relate: does this answer that question? Does this claim follow
from that one? So BERT gets a second, easy-to-generate pretraining task. Feed it two
sentences, and predict from the `[CLS]` vector whether B really came next.

> A: *The man went to the store.* B: *He bought a gallon of milk.* → **IsNext**
>
> A: *The man went to the store.* B: *Penguins are flightless birds.* → **NotNext**

The final model gets 97–98% right. The two tasks are trained together: every example has
masked words *and* a sentence pair.

::video[next-sentence]

**Why it mattered, and a twist.** The paper's own ablation (section 5) found NSP helped
on question answering and inference. A year later, **RoBERTa** retrained BERT more
carefully and found that dropping NSP did **no harm** once inputs were long, full
documents. Most of BERT's descendants left it out. Masking was the idea that lasted.

**Go deeper:** [BERT](/concepts/bert)

## 3.1 · Pretraining data

> **The paper says:** *"For the pre-training corpus we use the BooksCorpus (800M words)
> and English Wikipedia (2,500M words)."*

**In plain words.** 3.3 billion words, using only Wikipedia's running text (lists,
tables and headers stripped out). The paper insists on a **document-level** corpus, not
shuffled sentences, because next sentence prediction needs real neighbouring sentences,
and long passages teach long-range context.

**Why it mattered.** It was a lot of text for 2018. Five years later, models trained on
**trillions** of words showed that this was only the start.

## 3.2 · Fine-tuning: one model, many tasks

> **The paper says:** *"For each task, we simply plug in the task-specific inputs and
> outputs into BERT and fine-tune all the parameters end-to-end."*

**In plain words.** After pretraining, every task is a variation on one move: pack the
inputs as A (and B), and put a small head on the output vectors you need.

- **Classify** a sentence or pair (sentiment, entailment, paraphrase): a softmax on the
  `[CLS]` vector, $\text{softmax}(C\,W^\top)$. The paper: *"The only new parameters
  introduced during fine-tuning are classification layer weights"* $W \in
  \mathbb{R}^{K\times H}$.
- **Tag** each token (named entities): a classifier on **every** token's vector.
- **Answer questions** (SQuAD): question as A, passage as B. Learn a start vector $S$
  and an end vector $E$. The answer starts at the token that maximises $S\cdot T_i$ and
  ends at the token that maximises $E\cdot T_j$, with $j \ge i$.

And it's cheap: *"All of the results in the paper can be replicated in at most 1 hour on
a single Cloud TPU, or a few hours on a GPU, starting from the exact same pre-trained
model."*

::video[pretrain-finetune]

::figure[bert-heads]

**Why it mattered.** Pretraining costs days on special hardware, but it's done **once**,
by someone else. Anyone could download BERT and fine-tune it for their own task in an
afternoon. That's how BERT spread so quickly.

**Go deeper:** [BERT](/concepts/bert) · [Softmax](/concepts/softmax) ·
[Log-loss](/concepts/log-loss)

## 4 · Experiments: eleven records

> **The paper says:** *"Both BERT-BASE and BERT-LARGE outperform all systems on all tasks
> by a substantial margin, obtaining 4.5% and 7.0% respective average accuracy improvement
> over the prior state of the art."*

**In plain words.** The headline numbers:

- **GLUE**, a suite of nine sentence-understanding tasks: 80.5, up **7.7 points**. On its
  largest task, MNLI (does sentence B follow from A?), accuracy rose to 86.7%.
- **SQuAD v1.1**, find the answer span in a Wikipedia paragraph: F1 **93.2**, ahead of
  the best previous system and of the human score on the leaderboard.
- **SQuAD v2.0**, where some questions have no answer: F1 **83.1**, up 5.1 points.
- **SWAG**, choose the most plausible next sentence: 86.3%, **8.3 points** ahead of GPT.

The comparison that matters most is with GPT, which is the same size as BERT-base. On
GLUE, BERT-base averages 79.6 against GPT's 75.1. The paper notes that the two models
are *"nearly identical in terms of model architecture apart from the attention
masking"*.

**Why it mattered.** One recipe won across very different tasks. That convinced the
field that general-purpose pretraining, not task-specific design, was the way forward.

## 5 · Ablations: what really mattered

> **The paper says:** *"The LTR model performs worse than the MLM model on all tasks,
> with large drops on MRPC and SQuAD."*

**In plain words.** An ablation removes one ingredient and measures the damage. The paper
trains BERT-base three ways on exactly the same data:

| pretraining | MNLI | SQuAD F1 |
|---|---|---|
| masked LM + next sentence (BERT) | 84.4 | 88.5 |
| masked LM only | 83.9 | 87.9 |
| left-to-right LM only (like GPT) | 82.1 | **77.8** |

Reading in both directions is the big one: dropping it costs over **10 points** on
SQuAD. That makes sense. To decide where an answer *starts*, a token needs to see the
words after it, and a left-to-right model can't. Even adding a two-way LSTM on top
(the ELMo-style patch) only wins back part of the loss.

The paper also scales the model from 3 layers to 24, and accuracy improves every time,
even on a task with only 3,600 labelled examples. Bigger pretrained models help even
when the fine-tuning data is small.

**Why it mattered.** It's the evidence for the paper's title: the *deep bidirectional*
part is what made the difference.

**Go deeper:** [Masked language modeling](/concepts/masked-language-modeling) ·
[Causal masking](/concepts/causal-masking)

## 6 · Conclusion, and what came next

> **The paper says:** *"Our major contribution is further generalizing these findings to
> deep bidirectional architectures, allowing the same pre-trained model to successfully
> tackle a broad set of NLP tasks."*

**In plain words.** Pretraining was known to help. BERT showed how to pretrain a model
that reads **both ways**, and that one such model, lightly fine-tuned, beats specialists.

What happened next split the family in two:

- **Encoders kept improving.** RoBERTa (train longer, drop NSP), DistilBERT (smaller and
  faster), ALBERT, ELECTRA, DeBERTa and ModernBERT. They still power search ranking, text
  classification and the embedding step of retrieval-augmented generation (RAG).
- **Generation went to decoders.** GPT-2, GPT-3 and today's chat models went back to
  reading left to right, because a model that writes has to. But they kept BERT's
  recipe: pretrain on raw text first, then fine-tune for the job.

**Go deeper:** the whole route, from NumPy to the BERT you can train in 20 seconds, is
the [BERT track](/tracks/bert).
