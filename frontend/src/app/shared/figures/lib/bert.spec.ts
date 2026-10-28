import { corpusSentence, packPair, recover, wordPieces } from './bert';

const words = corpusSentence('zoe', 'jam');

describe('bert', () => {
  it('a hidden first name: guessed left to right, copied from the right both ways', () => {
    const hidden = new Set([0]);
    expect(recover(words, hidden, 0, false).probability).toBe(1 / 8);
    const both = recover(words, hidden, 0, true);
    expect(both.probability).toBe(1);
    expect(both.source).toBe(5);
  });

  it('the second copy is recoverable left to right, and nothing is when both are hidden', () => {
    expect(recover(words, new Set([5]), 5, false).source).toBe(0);
    const hidden = new Set([0, 5]);
    expect(recover(words, hidden, 0, true).probability).toBe(1 / 8);
    expect(recover(words, hidden, 5, true).reason).toContain('twin is hidden');
  });

  it('a left-to-right reader never sees what comes after', () => {
    const { visible } = recover(words, new Set([4]), 4, false);
    expect(visible).toEqual(words.map((_, s) => s < 4));
  });

  it('splits -ing and -ed off like WordPiece', () => {
    expect(wordPieces('He was Playing, then walked.')).toEqual([
      'he', 'was', 'play', '##ing', ',', 'then', 'walk', '##ed', '.',
    ]); // prettier-ignore
    expect(wordPieces('red sing')).toEqual(['red', 'sing']); // too short to split
  });

  it('packs [CLS] A [SEP] B [SEP] with segments and positions, within the limit', () => {
    const packed = packPair('the cat', 'it sat', 16);
    expect(packed.map((t) => t.token)).toEqual([
      '[CLS]',
      'the',
      'cat',
      '[SEP]',
      'it',
      'sat',
      '[SEP]',
    ]);
    expect(packed.map((t) => t.segment)).toEqual([0, 0, 0, 0, 1, 1, 1]);
    expect(packed.at(-1)?.position).toBe(6);
    expect(packPair('a '.repeat(30), 'b '.repeat(30), 16)).toHaveLength(16);
    expect(packPair('one sentence', '', 16).map((t) => t.token)).toEqual([
      '[CLS]',
      'one',
      'sentence',
      '[SEP]',
    ]);
  });
});
