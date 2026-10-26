import { Provider } from '@angular/core';
import { MARKED_EXTENSIONS, provideMarkdown } from 'ngx-markdown';

import { figureExtension, videoExtension } from './directive-extension';
import { mathExtension } from './math-extension';

/** Setup for `MarkdownView`; add to the `providers` of each lazy route that renders Markdown
 * so ngx-markdown and marked stay out of the initial bundle. */
export function provideMarkdownView(): Provider[] {
  return provideMarkdown({
    markedExtensions: [
      { provide: MARKED_EXTENSIONS, useValue: mathExtension, multi: true },
      { provide: MARKED_EXTENSIONS, useValue: figureExtension, multi: true },
      { provide: MARKED_EXTENSIONS, useValue: videoExtension, multi: true },
    ],
  });
}
