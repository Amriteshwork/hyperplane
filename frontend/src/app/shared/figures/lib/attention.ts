/** Scaled dot-product attention weights, and a hand-built sentence whose queries mean something. */

/** e^s / Σ e^s, safe for large scores; −Infinity gets weight exactly 0. */
export function softmax(scores: readonly number[]): number[] {
  const top = Math.max(...scores);
  const exps = scores.map((s) => Math.exp(s - top));
  const total = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / total);
}

function dot(a: readonly number[], b: readonly number[]): number {
  return a.reduce((sum, v, i) => sum + v * b[i], 0);
}

/**
 * Row t: how much token t takes from each token. `sharpness` multiplies every score, standing in
 * for "how big dot products get"; `causal` hides every token after t.
 */
export function attentionWeights(
  queries: readonly (readonly number[])[],
  keys: readonly (readonly number[])[],
  sharpness: number,
  causal: boolean,
): number[][] {
  return queries.map((q, t) =>
    softmax(keys.map((k, s) => (causal && s > t ? -Infinity : sharpness * dot(q, k)))),
  );
}

export interface Token {
  word: string;
  /** What the token offers: [living thing, describes/does something, pronoun, small word]. */
  key: readonly number[];
  /** What it looks for, in the same four features. */
  query: readonly number[];
  /** The query in plain words. */
  asks: string;
}

/** Real models learn their queries and keys; these are chosen by hand so they can be read. */
export const SENTENCE: readonly Token[] = [
  { word: 'the', key: [0, 0, 0, 1], query: [1, 0, 0, 0], asks: 'Which noun do I introduce?' },
  { word: 'animal', key: [1, 0, 0, 0], query: [0, 0, 0, 1], asks: 'What introduces me?' },
  { word: 'was', key: [0, 1, 0, 0.3], query: [1, 0, 0, 0], asks: 'Who is my subject?' },
  { word: 'tired', key: [0.2, 1, 0, 0], query: [1, 0, 0, 0], asks: 'Who is tired?' },
  { word: 'so', key: [0, 0, 0, 1], query: [0, 1, 0, 0], asks: 'What caused this?' },
  { word: 'it', key: [0.5, 0, 1, 0], query: [1, 0, 0, 0], asks: 'Which living thing am I?' },
  { word: 'slept', key: [0, 1, 0, 0], query: [1, 0, 1, 0], asks: 'Who slept?' },
];
