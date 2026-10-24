import type { MarkedExtension, Tokens } from 'marked';

/**
 * `::figure[name]` on its own line embeds an interactive figure (shared/figures). Like math, it
 * becomes a class-only placeholder holding the name as text, so the sanitiser keeps it, and
 * `MarkdownView` mounts the component into it afterwards.
 */
export const figureExtension: MarkedExtension = {
  extensions: [
    {
      name: 'figure',
      level: 'block',
      start: (src) => src.match(/^::figure\[/m)?.index,
      tokenizer(src) {
        const match = /^::figure\[([a-z0-9-]+)\][^\S\n]*(?:\n+|$)/.exec(src);
        if (match) return { type: 'figure', raw: match[0], name: match[1] };
        return undefined;
      },
      renderer: (token: Tokens.Generic) =>
        `<div class="figure-slot">${String(token['name'])}</div>`,
    },
  ],
};
