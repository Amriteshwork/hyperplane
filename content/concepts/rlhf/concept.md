---
title: RLHF
summary: People find it easier to say "A is better than B" than to write a perfect answer. Train a reward model on those votes, then nudge the chat model towards higher scores, but not too far, because a reward model can be fooled.
area: sequence-models
prerequisites: [chat-fine-tuning]
videos:
  votes-to-policy:
    title: From votes to a better model
    youtube: WEVWn3Chjcs
---

After [fine-tuning](/concepts/chat-fine-tuning), the model copies the style of its examples. But most
of what makes a reply good is hard to write down, and writing perfect replies is slow. Ask people an
easier question instead:

> **Which of these two replies is better?**

Anyone can answer that in seconds. **Reinforcement learning from human feedback** (RLHF) turns those
votes into a better model, in two steps. This is the recipe that InstructGPT (Ouyang et al., 2022)
used, and that ChatGPT followed.

::video[votes-to-policy]

### Step 1: turn votes into a score

Train a separate small model, the **reward model**, to give any reply a score. If people preferred
reply A to reply B, A's score should be higher:

$$P(A \text{ beats } B) = \sigma\big(r(A) - r(B)\big).$$

If you've read about [log-loss](/concepts/log-loss), you've seen this: it's logistic regression on the
*difference* between two scores. You be the rater, and watch the weights respond:

::figure[reward-votes]

### Step 2: push the model towards high scores

We can't backpropagate through "pick a token, then score the sentence", because picking isn't
differentiable. **REINFORCE** gets around that with a simple rule: sample some replies, score them, and
make each reply more or less likely in proportion to how far it beat the average.

$$\nabla \approx \big(r(\text{reply}) - \overline{r}\big)\ \nabla \log P(\text{reply}).$$

Replies that scored well become more likely. That's all "reinforcement learning" means here. (Real
systems use a steadier relative called PPO. The idea is the same.)

### Step 3: keep it on a leash

Here's the catch. The reward model is **a proxy**, trained only on the replies people happened to
compare. Push hard enough and the chat model will find replies that score well for the wrong reasons.
This is **Goodhart's law**: when a measure becomes a target, it stops being a good measure. So we charge
the model for drifting from where it started:

$$\text{maximise}\quad \mathbb{E}[r(\text{reply})] \;-\; \beta\ \mathrm{KL}\big(\text{policy} \,\Vert\, \text{base}\big).$$

This has an exact answer, which makes the trade-off easy to see:
$p(\text{reply}) \propto p_{\text{base}}(\text{reply}) \exp\!\big(r(\text{reply})/\beta\big)$. Slide
$\beta$. Raters said that a few extra polite words help, so the reward model learnt that more words
means a higher score. No rater ever saw the last reply:

::figure[rlhf-leash]

**Real quality rises, peaks, then collapses, while the reward score keeps climbing.** That's the most
important picture on this page. Gao et al. (2022) measured this same shape on real reward models.

### Does it work on a real model?

We tried it on the tiny chat model. Most of its examples were blunt, so it rarely said thanks. Votes
trained a reward model to prefer polite and correct answers. Then REINFORCE ran for just 30 steps:

| | says "thanks" | names the right food |
|---|---|---:|
| before RLHF | 8% | 88% |
| after, tight leash ($\beta = 1$) | 24% | 93% |
| after, loose leash ($\beta = 0.1$) | 78% | 84% |

The behaviour moved towards the votes, and the leash controlled how far. With the loose leash, the
share of right answers slipped: the small reward model valued the polite ending more than it was able
to check the facts. The exact percentages change with the seed, and our model is tiny, but the
direction is the point.

### The failures you've met in real chatbots

Now some familiar behaviour has an explanation:

- **Sycophancy.** People often prefer replies that agree with them, so a model trained on their votes
  learns to agree. Anthropic documented this in 2023.
- **Over-long, confident replies.** If raters tend to prefer long, sure-sounding answers, that's what
  the reward model learns to score.
- **Rater limits.** Votes can only reward what raters can check. A plausible wrong answer may beat an
  honest "I don't know".

Modern systems add more on top. For example, Constitutional AI has AI models give some of the
feedback, using written principles. The core loop, of comparing, scoring and shifting, stays recognisable.

**What the code shows.** The NumPy file fits the reward from votes, checks that REINFORCE lands on the
exact answer above, and asserts that a moderate $\beta$ beats the base model while an extreme one is
worse. The PyTorch file runs the full pipeline on the tiny GPT in about 40 seconds and produces the table.
