"""RLHF in PyTorch: votes train a reward model, and REINFORCE moves a tiny chat model towards what the votes liked."""

import copy
import random

import torch
import torch.nn.functional as F

# @a:reuse
from chat_fine_tuning_pytorch import CONTEXT, EXTRA, STOP, chat_tensors, make_chat, pretrain
from language_modeling_pytorch import NAMES, make_corpus
from tokenization_pytorch import build_vocab, decode, encode
from transformer_pytorch import TinyGPT

#: All four letters long, so every prompt in a batch lines up without padding.
FOODS = ["figs", "rice", "nuts", "soup", "kale", "eggs"]
LIMIT = 14


def prompt_for(pick):
    """A fact and a question about it, and the right food."""
    name, food = pick.choice(NAMES), pick.choice(FOODS)
    return f"{name} eats {food}.<u>what does {name} eat?<a>", food


# @a:reward-model
class RewardModel(torch.nn.Module):
    """A conversation → one score: the same tiny GPT, reading out one number instead of next-token scores."""

    def __init__(self, V):
        super().__init__()
        self.net = TinyGPT(V, context=CONTEXT)
        self.net.unembed = torch.nn.Linear(self.net.unembed.in_features, 1)

    def forward(self, ids):
        return self.net(ids)[..., -1, 0]


# @a:votes
def train_reward(vocab, steps=150):
    """Each vote says A beat B. Train on σ(r(A) − r(B)): log-loss on the difference between two scores."""
    torch.manual_seed(0)
    reward = RewardModel(len(vocab))
    optimizer = torch.optim.AdamW(reward.parameters(), lr=3e-3)
    pick = random.Random(0)

    def conversation(prompt, reply):
        return encode((prompt + reply + STOP).ljust(len(prompt) + LIMIT), vocab)

    for _ in range(steps):
        winners, losers = [], []
        for _ in range(16):
            prompt, food = prompt_for(pick)
            wrong = pick.choice([f for f in FOODS if f != food])
            ranked = [f"{food}, thanks.", f"{food}.", f"{wrong}, thanks.", f"{wrong}."]  # best first
            i, j = sorted(pick.sample(range(4), 2))
            winners.append(conversation(prompt, ranked[i]))
            losers.append(conversation(prompt, ranked[j]))
        loss = F.softplus(reward(torch.stack(losers)) - reward(torch.stack(winners))).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return reward


# @a:sample
@torch.no_grad()
def sample_replies(model, prompts, vocab):
    """Draw whole replies at once. After the stop token, everything is padding."""
    ids, stop, blank = prompts, vocab[STOP], vocab[" "]
    alive = torch.ones(len(ids), dtype=torch.bool)
    for _ in range(LIMIT):
        token = torch.multinomial(torch.softmax(model(ids)[:, -1], -1), 1)[:, 0]
        ids = torch.cat([ids, torch.where(alive, token, blank)[:, None]], dim=1)
        alive = alive & (token != stop)
    return ids


# @a:log-prob
def reply_log_prob(model, ids, start, stop):
    """log P(reply) under `model`: one pass over the whole conversation, adding up the reply's tokens."""
    picked = torch.log_softmax(model(ids[:, :-1]), -1).gather(2, ids[:, 1:, None])[..., 0]
    reply = ids[:, start:]
    # Count up to and including the stop token: padding was never chosen by the model.
    alive = (reply == stop).cumsum(1).roll(1, 1).index_fill(1, torch.tensor([0]), 0) == 0
    return (picked[:, start - 1 :] * alive).sum(1)


# @a:kl
def rlhf_step(policy, base, reward, vocab, optimizer, pick, beta=0.1, batch=32):
    """REINFORCE: raise the odds of replies that beat the average, after charging them for drifting from base."""
    prompts = torch.stack([encode(prompt_for(pick)[0], vocab) for _ in range(batch)])
    ids = sample_replies(policy, prompts, vocab)
    log_prob = reply_log_prob(policy, ids, prompts.shape[1], vocab[STOP])
    with torch.no_grad():
        drift = log_prob - reply_log_prob(base, ids, prompts.shape[1], vocab[STOP])
        score = reward(ids) - beta * drift
    loss = -((score - score.mean()) * log_prob).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# @a:measure
@torch.no_grad()
def behaviour(model, vocab, n=100):
    """Sample replies to unseen prompts: how many say thanks, and how many name the right food?"""
    torch.manual_seed(7)
    pick = random.Random(321)
    asked = [prompt_for(pick) for _ in range(n)]
    prompts = torch.stack([encode(p, vocab) for p, _ in asked])
    said = [decode(row, vocab) for row in sample_replies(model, prompts, vocab)[:, prompts.shape[1] :]]
    polite = sum("thanks" in s for s in said) / n
    right = sum(s.startswith(food) for s, (_, food) in zip(said, asked)) / n
    return polite, right, said[0].strip()


# @a:demo
if __name__ == "__main__":
    # @a:start
    vocab = build_vocab(sorted(set(make_corpus(1000) + EXTRA + ",thanks")))
    pick = random.Random(0)
    # Most fine-tuning examples are blunt, so the starting model rarely says thanks.
    chats = [make_chat(pick, polite=i % 8 == 0) for i in range(200)]
    x, y = chat_tensors(chats, vocab)
    base = pretrain(vocab, steps=150)
    optimizer = torch.optim.AdamW(base.parameters(), lr=3e-3)
    torch.manual_seed(1)
    for _ in range(100):
        batch = torch.randint(len(x), (16,))
        loss = F.cross_entropy(base(x[batch]).reshape(-1, len(vocab)), y[batch].reshape(-1), ignore_index=-100)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    before = behaviour(base, vocab)
    print(f"before RLHF: polite {before[0]:.0%}, right food {before[1]:.0%}, e.g. {before[2]!r}")

    # @a:reward
    reward = train_reward(vocab)
    prompt, food = prompt_for(pick)

    def score(reply):
        with torch.no_grad():
            return reward(encode((prompt + reply + STOP).ljust(len(prompt) + LIMIT), vocab)).item()

    print(f"reward: polite {score(f'{food}, thanks.'):+.1f}, blunt {score(f'{food}.'):+.1f}, "
          f"wrong food {score('jam.'):+.1f}")
    assert score(f"{food}, thanks.") > score(f"{food}.")

    # @a:improve
    results = {}
    for beta in (1.0, 0.1):  # a tight leash on the base model, then a loose one
        policy = copy.deepcopy(base)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
        torch.manual_seed(2)
        for _ in range(30):
            rlhf_step(policy, base, reward, vocab, optimizer, pick, beta=beta)
        results[beta] = behaviour(policy, vocab)
        print(f"after RLHF, β={beta}: polite {results[beta][0]:.0%}, right food {results[beta][1]:.0%}, "
              f"e.g. {results[beta][2]!r}")
    assert results[0.1][0] > before[0] + 0.3  # the behaviour moved towards the votes
