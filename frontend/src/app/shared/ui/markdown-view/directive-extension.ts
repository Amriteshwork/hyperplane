import type { MarkedExtension, Tokens } from 'marked';

/**
 * `::<kind>[name]` on its own line becomes a class-only placeholder holding the name as text,
 * so the sanitiser keeps it, and `MarkdownView` mounts a component into it afterwards.
 */
function blockDirective(kind: string): MarkedExtension {
  const pattern = new RegExp(`^::${kind}\\[([a-z0-9-]+)\\][^\\S\\n]*(?:\\n+|$)`);
  return {
    extensions: [
      {
        name: kind,
        level: 'block',
        start: (src) => src.match(new RegExp(`^::${kind}\\[`, 'm'))?.index,
        tokenizer(src) {
          const match = pattern.exec(src);
          if (match) return { type: kind, raw: match[0], name: match[1] };
          return undefined;
        },
        renderer: (token: Tokens.Generic) =>
          `<div class="${kind}-slot">${String(token['name'])}</div>`,
      },
    ],
  };
}

/** `::figure[name]`: an interactive figure from shared/figures (ADR 0020). */
export const figureExtension = blockDirective('figure');

/** `::video[name]`: a Manim animation, resolved from the page's own video list (ADR 0022). */
export const videoExtension = blockDirective('video');
