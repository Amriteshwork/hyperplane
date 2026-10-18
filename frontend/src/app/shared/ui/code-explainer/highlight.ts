import type { HighlighterCore } from 'shiki/core';

/** One rendered token; `style` carries `--shiki-light` / `--shiki-dark` colours. */
export interface CodeToken {
  readonly content: string;
  readonly style: Record<string, string> | null;
}
export type CodeLine = readonly CodeToken[];

/** Grammars bundled on demand; extend when a pack ships code in another language. */
const LANGUAGES = {
  python: () => import('shiki/langs/python.mjs'),
} as const;
export type CodeLanguage = keyof typeof LANGUAGES;

let highlighter: Promise<HighlighterCore> | null = null;

/** Shiki is large, so it is loaded the first time code is shown and then reused. */
function getHighlighter(): Promise<HighlighterCore> {
  highlighter ??= Promise.all([import('shiki/core'), import('shiki/engine/javascript')]).then(
    ([{ createHighlighterCore }, { createJavaScriptRegexEngine }]) =>
      createHighlighterCore({
        themes: [import('shiki/themes/github-light.mjs'), import('shiki/themes/github-dark.mjs')],
        langs: Object.values(LANGUAGES).map((load) => load()),
        engine: createJavaScriptRegexEngine(),
      }),
  );
  return highlighter;
}

/** Unhighlighted lines, used until Shiki is ready (and in tests). */
export function plainLines(code: string): CodeLine[] {
  return code.split('\n').map((line) => [{ content: line, style: null }]);
}

export async function highlightLines(code: string, lang: CodeLanguage): Promise<CodeLine[]> {
  const shiki = await getHighlighter();
  const { tokens } = shiki.codeToTokens(code, {
    lang,
    themes: { light: 'github-light', dark: 'github-dark' },
    defaultColor: false,
  });
  return tokens.map((line) =>
    line.map((token) => ({ content: token.content, style: token.htmlStyle ?? null })),
  );
}
