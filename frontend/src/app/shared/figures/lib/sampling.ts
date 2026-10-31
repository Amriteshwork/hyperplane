/** Text generation's figures, as plain functions: sampling knobs, beam search and the KV cache's cost. */

export interface Knobs {
  temperature: number;
  topK: number | null;
  topP: number | null;
}

/** Scores → the distribution to draw from: temperature first, then cut the tail. */
export function nextTokenProbs(logits: readonly number[], knobs: Knobs): number[] {
  const scaled = logits.map((z) => z / Math.max(knobs.temperature, 1e-4));
  const top = Math.max(...scaled);
  const raw = scaled.map((z) => Math.exp(z - top));
  const total = raw.reduce((a, b) => a + b, 0);
  const p = raw.map((v) => v / total);
  const order = p.map((_, i) => i).sort((a, b) => p[b] - p[a]);
  const keep = p.map(() => true);
  let before = 0;
  order.forEach((token, rank) => {
    if (knobs.topK !== null && rank >= knobs.topK) keep[token] = false;
    if (knobs.topP !== null && before >= knobs.topP) keep[token] = false;
    before += p[token];
  });
  const kept = p.reduce((sum, v, i) => sum + (keep[i] ? v : 0), 0);
  return p.map((v, i) => (keep[i] ? v / kept : 0));
}

/** Which token a uniform draw in [0, 1) lands on. */
export function drawToken(p: readonly number[], u: number): number {
  let sum = 0;
  for (let i = 0; i < p.length; i++) {
    sum += p[i];
    if (u < sum) return i;
  }
  // Rounding can leave `u` just past the total: fall back to the last token with any chance.
  return p.map((v, i) => (v > 0 ? i : -1)).reduce((a, b) => Math.max(a, b));
}

/** The next-word odds after a word, for the beam search figure's small tree. */
export const TREE: Readonly<Record<string, Readonly<Record<string, number>>>> = {
  start: { the: 0.5, a: 0.4, one: 0.1 },
  the: { cat: 0.4, dog: 0.35, end: 0.25 },
  a: { cat: 0.9, dog: 0.1 },
  one: { cat: 1 },
};

export interface Beam {
  words: readonly string[];
  probability: number;
}

/** The beams kept after each step: grow every kept path by every word, keep the best `width`. */
export function beamSteps(width: number, steps = 2): Beam[][] {
  const out: Beam[][] = [[{ words: [], probability: 1 }]];
  for (let s = 0; s < steps; s++) {
    const grown = out[s].flatMap((beam) =>
      Object.entries(TREE[beam.words.at(-1) ?? 'start'] ?? {}).map(([word, p]) => ({
        words: [...beam.words, word],
        probability: beam.probability * p,
      })),
    );
    out.push(grown.sort((a, b) => b.probability - a.probability).slice(0, width));
  }
  return out;
}

/** Attention scores computed to write `tokens` tokens: rerunning the model, or with the KV cache. */
export function attentionWork(tokens: number): { rerun: number; cached: number } {
  let rerun = 0;
  let cached = 0;
  for (let t = 1; t <= tokens; t++) {
    rerun += t * t;
    cached += t;
  }
  return { rerun, cached };
}
