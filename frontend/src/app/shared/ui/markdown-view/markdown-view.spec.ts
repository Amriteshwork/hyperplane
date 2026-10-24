import { Component } from '@angular/core';
import { TestBed } from '@angular/core/testing';

import { FIGURES } from '../../figures/figure-registry';
import { MarkdownView } from './markdown-view';
import { provideMarkdownView } from './provide-markdown-view';

@Component({ selector: 'app-stub-figure', template: '<p class="stub">interactive</p>' })
class StubFigure {}

async function render(content: string): Promise<HTMLElement> {
  TestBed.configureTestingModule({
    providers: [
      provideMarkdownView(),
      { provide: FIGURES, useValue: { stub: async () => StubFigure } },
    ],
  });
  const fixture = TestBed.createComponent(MarkdownView);
  fixture.componentRef.setInput('content', content);
  await fixture.whenStable();
  await new Promise((resolve) => setTimeout(resolve));
  await fixture.whenStable();
  return fixture.nativeElement;
}

describe('MarkdownView figures', () => {
  it('mounts a registered figure where its directive was', async () => {
    const el = await render('Before.\n\n::figure[stub]\n\nAfter.');
    const slot = el.querySelector('.figure-slot');
    expect(slot?.querySelector('.stub')?.textContent).toBe('interactive');
    expect(el.textContent).not.toContain('::figure');
  });

  it('says so for an unknown or inherited name instead of failing', async () => {
    const el = await render('::figure[missing]\n\n::figure[constructor]');
    const slots = [...el.querySelectorAll('.figure-slot')].map((s) => s.textContent);
    expect(slots).toEqual([
      '(Figure “missing” is not available.)',
      '(Figure “constructor” is not available.)',
    ]);
  });
});
