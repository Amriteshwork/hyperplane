import type { MarkedExtension, Tokens } from 'marked';

/**
 * Shields `$…$` / `$$…$$` from Markdown (so `x_i` isn't turned into emphasis) by emitting the
 * TeX as escaped text in `.math` placeholders. KaTeX renders them after Angular's sanitiser has
 * run, so math never needs `SecurityContext.NONE`. Class names only: the sanitiser drops `data-*`.
 */
export const mathExtension: MarkedExtension = {
  extensions: [
    {
      name: 'mathBlock',
      level: 'block',
      start: (src) => src.match(/\$\$/)?.index,
      tokenizer(src) {
        const match = /^\$\$([\s\S]+?)\$\$(?:\n+|$)/.exec(src);
        if (match) return { type: 'mathBlock', raw: match[0], text: match[1].trim() };
        return undefined;
      },
      renderer: (token) => `<div class="math math-display">${escape(text(token))}</div>`,
    },
    {
      name: 'mathInline',
      level: 'inline',
      start: (src) => src.match(/\$/)?.index,
      tokenizer(src) {
        const match = /^\$(?!\s)((?:\\.|[^\\$\n])+?)(?<!\s)\$/.exec(src);
        if (match) return { type: 'mathInline', raw: match[0], text: match[1] };
        return undefined;
      },
      renderer: (token) => `<span class="math">${escape(text(token))}</span>`,
    },
  ],
};

function text(token: Tokens.Generic): string {
  return String(token['text'] ?? '');
}

function escape(value: string): string {
  return value.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

/** Renders every placeholder under `root`; KaTeX is loaded only when math is present. */
export async function renderMath(root: HTMLElement): Promise<void> {
  const nodes = root.querySelectorAll<HTMLElement>('.math:not([data-rendered])');
  if (!nodes.length) return;
  const { default: katex } = await import('katex');
  for (const node of nodes) {
    katex.render(node.textContent ?? '', node, {
      displayMode: node.classList.contains('math-display'),
      throwOnError: false,
    });
    node.dataset['rendered'] = '';
  }
}
