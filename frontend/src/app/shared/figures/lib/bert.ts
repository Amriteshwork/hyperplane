/** BERT's figures, as plain functions: which hidden word a reader can recover, and packed inputs. */

/** The same generated corpus as the Language-modeling page's code. */
export const NAMES = ['ann', 'bob', 'eve', 'max', 'zoe', 'ian', 'kim', 'raj'] as const;
export const FOODS = ['figs', 'plums', 'rice', 'nuts', 'soup', 'kale', 'eggs', 'jam'] as const;

export interface Word {
  text: string;
  /** Names and foods are random; every other word is fixed by the sentence pattern. */
  kind: 'name' | 'food' | 'fixed';
  /** The other copy of a name or food, which repeats it. */
  twin: number | null;
}

/** "zoe likes jam , so zoe eats jam .": words 5 and 7 copy words 0 and 2. */
export function corpusSentence(name: string, food: string): Word[] {
  const fixed = (text: string): Word => ({ text, kind: 'fixed', twin: null });
  return [
    { text: name, kind: 'name', twin: 5 },
    fixed('likes'),
    { text: food, kind: 'food', twin: 7 },
    fixed(','),
    fixed('so'),
    { text: name, kind: 'name', twin: 0 },
    fixed('eats'),
    { text: food, kind: 'food', twin: 2 },
    fixed('.'),
  ];
}

export interface Recovery {
  /** Which positions this reader may look at while filling position `at`. */
  visible: boolean[];
  /** The position the answer is copied from, if any. */
  source: number | null;
  /** Chance of getting the word right. */
  probability: number;
  reason: string;
}

/**
 * How a reader fills the hidden word at `at`. A left-to-right reader sees only earlier
 * positions; a bidirectional one sees every position that isn't hidden itself.
 */
export function recover(
  words: readonly Word[],
  hidden: ReadonlySet<number>,
  at: number,
  bidirectional: boolean,
): Recovery {
  const visible = words.map((_, s) => s !== at && !hidden.has(s) && (bidirectional || s < at));
  const word = words[at];
  if (word.kind === 'fixed')
    return { visible, source: null, probability: 1, reason: 'The pattern fixes this word.' };
  const choices = word.kind === 'name' ? NAMES.length : FOODS.length;
  const twin = word.twin;
  if (twin !== null && visible[twin])
    return {
      visible,
      source: twin,
      probability: 1,
      reason: `It copies “${words[twin].text}” from position ${twin}.`,
    };
  const why =
    twin !== null && hidden.has(twin)
      ? 'Its twin is hidden too'
      : bidirectional
        ? 'Nothing gives it away'
        : 'Its twin is to the right, out of sight';
  return {
    visible,
    source: null,
    probability: 1 / choices,
    reason: `${why}: a guess, 1 in ${choices}.`,
  };
}

/** A toy WordPiece: lowercase words and punctuation, with "-ing" and "-ed" split off as "##ing". */
export function wordPieces(text: string): string[] {
  const words = text.toLowerCase().match(/[a-z0-9']+|[^\sa-z0-9']/g) ?? [];
  return words.flatMap((w) => {
    const suffix = ['ing', 'ed'].find((s) => w.endsWith(s) && w.length >= s.length + 3);
    return suffix ? [w.slice(0, -suffix.length), `##${suffix}`] : [w];
  });
}

export interface InputToken {
  token: string;
  /** 0 for sentence A (and [CLS]), 1 for sentence B. */
  segment: 0 | 1;
  position: number;
}

/** [CLS] A [SEP] B [SEP], with each sentence cut so the whole input fits in `limit` tokens. */
export function packPair(a: string, b: string, limit: number): InputToken[] {
  const room = Math.floor((limit - 3) / 2);
  const first = wordPieces(a).slice(0, room);
  const second = wordPieces(b).slice(0, limit - 3 - first.length);
  const tokens: [string, 0 | 1][] = [
    ['[CLS]', 0],
    ...first.map((t): [string, 0] => [t, 0]),
    ['[SEP]', 0],
  ];
  if (second.length) tokens.push(...second.map((t): [string, 1] => [t, 1]), ['[SEP]', 1]);
  return tokens.map(([token, segment], position) => ({ token, segment, position }));
}
