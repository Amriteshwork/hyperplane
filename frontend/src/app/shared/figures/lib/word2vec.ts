/** Word embeddings' figures: live skip-gram training, plus nearest words and analogies on real vectors. */
import { Rng, normal } from './random';
import { REAL_VECTORS } from './real-vectors.data';

export const GROUPS = {
  animal: ['cat', 'dog', 'fox', 'owl'],
  food: ['rice', 'soup', 'figs', 'jam'],
  vehicle: ['car', 'bus', 'tram', 'van'],
} as const;
export type Group = keyof typeof GROUPS;

const PATTERNS: Readonly<Record<Group, readonly string[]>> = {
  animal: ['the {} sleeps', 'a {} runs fast', 'the {} hunts at night', 'my {} chases birds'],
  food: ['we cook {}', 'hot {} tastes good', 'i eat {} at lunch', 'fresh {} on my plate'],
  vehicle: ['the {} stops here', 'a {} needs fuel', 'we drive the {}', 'my {} has wheels'],
};

export const dot = (a: readonly number[], b: readonly number[]): number =>
  a.reduce((sum, v, i) => sum + v * b[i], 0);
export const cosine = (a: readonly number[], b: readonly number[]): number =>
  dot(a, b) / Math.sqrt(dot(a, a) * dot(b, b));
const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

export interface SkipGram {
  vocab: string[];
  /** Each word as a centre: these are the vectors we keep. */
  E: number[][];
  /** Each word as a neighbour. */
  C: number[][];
  pairs: [number, number][];
}

/** The same corpus as the page's code: words in a group share sentence patterns, and nothing else. */
export function createSkipGram(rng: Rng, sentences = 800, window = 2): SkipGram {
  const groups = Object.keys(GROUPS) as Group[];
  const texts = Array.from({ length: sentences }, () => {
    const group = groups[Math.floor(rng() * groups.length)];
    const pattern = PATTERNS[group][Math.floor(rng() * PATTERNS[group].length)];
    const word = GROUPS[group][Math.floor(rng() * GROUPS[group].length)];
    return pattern.replace('{}', word).split(' ');
  });
  const vocab = [...new Set(texts.flat())].sort();
  const id = new Map(vocab.map((w, i) => [w, i]));
  const pairs: [number, number][] = [];
  for (const words of texts)
    words.forEach((w, i) => {
      for (let j = Math.max(0, i - window); j < Math.min(words.length, i + window + 1); j++)
        if (j !== i) pairs.push([id.get(w)!, id.get(words[j])!]);
    });
  const table = () => vocab.map(() => [normal(rng) * 0.5, normal(rng) * 0.5]);
  return { vocab, E: table(), C: table(), pairs };
}

/** One SGD step on one pair: pull the real neighbour in, push `k` random words away. */
export function skipGramStep(model: SkipGram, rng: Rng, lr = 0.05, k = 3): void {
  const [center, context] = model.pairs[Math.floor(rng() * model.pairs.length)];
  const u = model.E[center];
  const move = (v: number[], label: number) => {
    const pull = label - sigmoid(dot(u, v));
    const du = [pull * v[0], pull * v[1]];
    v[0] += lr * pull * u[0];
    v[1] += lr * pull * u[1];
    return du;
  };
  const total = [0, 0];
  const add = (d: number[]) => {
    total[0] += d[0];
    total[1] += d[1];
  };
  add(move(model.C[context], 1));
  for (let n = 0; n < k; n++) add(move(model.C[Math.floor(rng() * model.vocab.length)], 0));
  u[0] += lr * total[0];
  u[1] += lr * total[1];
}

/** Mean same-group cosine minus mean different-group cosine: how far the groups have separated. */
export function groupSeparation(model: SkipGram): number {
  const groups = Object.values(GROUPS) as readonly (readonly string[])[];
  const vector = (w: string) => model.E[model.vocab.indexOf(w)];
  let same = 0;
  let sameN = 0;
  let other = 0;
  let otherN = 0;
  groups.forEach((g, i) =>
    groups.forEach((h, j) => {
      for (const a of g)
        for (const b of h) {
          if (a === b) continue;
          const c = cosine(vector(a), vector(b));
          if (i === j) {
            same += c;
            sameN++;
          } else {
            other += c;
            otherN++;
          }
        }
    }),
  );
  return same / sameN - other / otherN;
}

export const WORDS = REAL_VECTORS.words as unknown as Readonly<Record<string, readonly number[]>>;
export const WORD_LIST = Object.keys(WORDS).sort();
export const ANALOGIES = REAL_VECTORS.analogies as unknown as readonly (readonly [
  string,
  string,
  string,
])[];

export interface Near {
  word: string;
  cos: number;
}

function ranked(target: readonly number[], skip: readonly string[], n: number): Near[] {
  return WORD_LIST.filter((w) => !skip.includes(w))
    .map((word) => ({ word, cos: cosine(WORDS[word], target) }))
    .sort((a, b) => b.cos - a.cos)
    .slice(0, n);
}

export const nearestWords = (word: string, n = 5): Near[] => ranked(WORDS[word], [word], n);

/** a : b :: c : ?, computed as b − a + c, leaving the three inputs out of the search. */
export function solveAnalogy(a: string, b: string, c: string, n = 4): Near[] {
  const target = WORDS[b].map((v, i) => v - WORDS[a][i] + WORDS[c][i]);
  return ranked(target, [a, b, c], n);
}

/** The words projected onto their two main directions (PCA by power iteration), for the map. */
export function pca2d(words: readonly string[]): Record<string, [number, number]> {
  const dims = WORDS[words[0]].length;
  const mean = Array.from(
    { length: dims },
    (_, d) => words.reduce((sum, w) => sum + WORDS[w][d], 0) / words.length,
  );
  const rows = words.map((w) => WORDS[w].map((v, d) => v - mean[d]));
  const axes: number[][] = [];
  for (let axis = 0; axis < 2; axis++) {
    let v = Array.from({ length: dims }, (_, d) => Math.cos(d + axis));
    for (let iter = 0; iter < 80; iter++) {
      const next = new Array<number>(dims).fill(0);
      for (const row of rows) {
        const s = dot(row, v);
        row.forEach((x, d) => (next[d] += s * x));
      }
      for (const done of axes) {
        const s = dot(next, done);
        done.forEach((x, d) => (next[d] -= s * x));
      }
      const len = Math.sqrt(dot(next, next));
      v = next.map((x) => x / len);
    }
    axes.push(v);
  }
  return Object.fromEntries(
    words.map((w, i): [string, [number, number]] => [
      w,
      [dot(rows[i], axes[0]), dot(rows[i], axes[1])],
    ]),
  );
}

export interface TokenRow {
  id: number;
  text: string;
  strip: readonly number[];
  near: readonly { text: string; cos: number }[];
}
export const TOKEN_ROWS: readonly TokenRow[] = REAL_VECTORS.tokens;
export const TOKEN_SENTENCES: readonly { text: string; ids: readonly number[] }[] =
  REAL_VECTORS.sentences;
