/**
 * Results measured by running content/concepts/<id>/code/pytorch.py. They're typed in rather than
 * computed live, because each took seconds of training. Rerun that file to reproduce them.
 */

/** The position of a row's largest weight. */
export function argmax(row: readonly number[]): number {
  return row.reduce((best, v, i) => (v > row[best] ? i : best), 0);
}

/** text-generation: 200 sentences at each temperature. */
export const TEMPERATURE_RUNS = [
  { temperature: 0.01, valid: 1.0, distinct: 1, example: 'max likes soup, so max eats soup.' },
  { temperature: 0.3, valid: 0.98, distinct: 32, example: 'max likes plums, so max eats plums.' },
  { temperature: 0.6, valid: 0.86, distinct: 64, example: 'raj likes plums, so raj eats plums.' },
  {
    temperature: 1.0,
    valid: 0.58,
    distinct: 117,
    example: 'kim likes plums, soso max eats plums.',
  },
  {
    temperature: 1.5,
    valid: 0.14,
    distinct: 188,
    example: 'kim likes plums, soso max eats plums.',
  },
  { temperature: 2.5, valid: 0, distinct: 199, example: 'kkim liknutss.' },
] as const;

/** seq2seq-attention: share of unseen lists translated exactly, by list length. */
export const BOTTLENECK = {
  lengths: [1, 2, 3, 4, 5, 6, 8, 10, 12],
  plain: [1, 1, 1, 0.94, 0.82, 0.68, 0.26, 0.03, 0],
  attention: [1, 1, 1, 0.99, 1, 1, 0.98, 0.91, 0.7],
} as const;

/** seq2seq-attention: the trained model's weights, one row per German word. */
export const LOOKBACK = {
  english: ['figs', 'jam', 'rice', 'kale', 'eggs', 'soup'],
  german: ['feigen', 'marmelade', 'reis', 'grünkohl', 'eier', 'suppe'],
  weights: [
    [0.87, 0.09, 0.01, 0.01, 0.01, 0.01],
    [0.16, 0.71, 0.11, 0.01, 0.01, 0.0],
    [0.0, 0.22, 0.63, 0.13, 0.01, 0.01],
    [0.0, 0.0, 0.29, 0.58, 0.1, 0.02],
    [0.0, 0.0, 0.0, 0.25, 0.56, 0.18],
    [0.0, 0.0, 0.0, 0.0, 0.17, 0.83],
  ],
} as const;

/** encoder-decoder: last decoder block's cross-attention, averaged over heads. Last row is [EOS]. */
export const CROSS_ATTENTION = {
  english: ['eve', 'likes', 'soup', ',', 'so', 'bob', 'eats', 'plums', '.'],
  german: ['eve', 'mag', 'suppe', ',', 'also', 'isst', 'bob', 'pflaumen', '.'],
  weights: [
    [0.75, 0.04, 0.02, 0.13, 0.0, 0.05, 0.01, 0.0, 0.0],
    [0.36, 0.12, 0.26, 0.04, 0.03, 0.09, 0.03, 0.02, 0.04],
    [0.0, 0.0, 0.96, 0.0, 0.0, 0.0, 0.0, 0.03, 0.0],
    [0.57, 0.09, 0.11, 0.03, 0.05, 0.03, 0.03, 0.01, 0.09],
    [0.26, 0.1, 0.18, 0.04, 0.07, 0.04, 0.1, 0.14, 0.06],
    [0.13, 0.1, 0.0, 0.01, 0.45, 0.01, 0.1, 0.06, 0.15],
    [0.0, 0.01, 0.0, 0.04, 0.12, 0.16, 0.37, 0.07, 0.23],
    [0.0, 0.0, 0.11, 0.11, 0.04, 0.0, 0.04, 0.64, 0.05],
    [0.01, 0.01, 0.0, 0.04, 0.2, 0.11, 0.06, 0.12, 0.46],
  ],
} as const;

/** llm-training: validation loss for each run, all from one seed. */
export const WARMUP_RUNS = [
  { label: 'lr 0.01, constant', loss: 0.329, note: 'a safe learning rate' },
  { label: 'lr 0.1, constant', loss: 1.577, note: 'too fast, and it never recovers' },
  { label: 'lr 0.1, clipping', loss: 1.548, note: "clipping alone doesn't save it" },
  { label: 'lr 0.1, warmup + cosine', loss: 0.312, note: 'the same peak, reached gradually' },
] as const;

/** chat-fine-tuning: replies from each model, and the share of 50 unseen chats answered exactly. */
export const CHAT_RUNS = [
  {
    label: 'Base model',
    rate: 0,
    reply: ' p. ranu ligs. r',
    note: 'It keeps continuing text and never stops. Nothing has told it this is a question.',
  },
  {
    label: 'No pretraining',
    rate: 0.58,
    reply: null,
    note: 'The same 16 chats and 80 steps, from random weights.',
  },
  {
    label: 'Pretrained, then fine-tuned',
    rate: 0.92,
    reply: 'eggs.|',
    note: 'The right food, then the stop token.',
  },
] as const;
