/** Parameter counts of a GPT-2-style transformer (learned positions, biases, tied unembedding). */

export interface GptShape {
  vocab: number;
  d: number;
  layers: number;
  context: number;
}

export interface GptCounts {
  tokenEmbedding: number;
  positions: number;
  /** All blocks together: W_Q, W_K, W_V, W_O and their biases. */
  attention: number;
  /** All blocks together: d → 4d → d, with biases. */
  mlp: number;
  /** Every layer norm's γ and β, including the final one. */
  norms: number;
  total: number;
}

export function parameterCounts({ vocab, d, layers, context }: GptShape): GptCounts {
  const tokenEmbedding = vocab * d; // also the unembedding: GPT-2 reuses the same matrix
  const positions = context * d;
  const attention = layers * (4 * d * d + 4 * d);
  const mlp = layers * (8 * d * d + 5 * d);
  const norms = layers * 4 * d + 2 * d;
  const total = tokenEmbedding + positions + attention + mlp + norms;
  return { tokenEmbedding, positions, attention, mlp, norms, total };
}

export const PRESETS: readonly { label: string; shape: GptShape; heads: number }[] = [
  { label: 'GPT-2 small', shape: { vocab: 50257, d: 768, layers: 12, context: 1024 }, heads: 12 },
  { label: 'GPT-2 XL', shape: { vocab: 50257, d: 1600, layers: 48, context: 1024 }, heads: 25 },
  { label: 'GPT-3', shape: { vocab: 50257, d: 12288, layers: 96, context: 2048 }, heads: 96 },
];
