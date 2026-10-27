import { Component } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { Router, provideRouter } from '@angular/router';

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

describe('MarkdownView videos and links', () => {
  async function renderWith(content: string) {
    TestBed.configureTestingModule({
      providers: [provideMarkdownView(), provideRouter([]), { provide: FIGURES, useValue: {} }],
    });
    const fixture = TestBed.createComponent(MarkdownView);
    fixture.componentRef.setInput('content', content);
    fixture.componentRef.setInput('videos', {
      ready: { title: 'Soft lookup', youtubeId: 'dQw4w9WgXcQ' },
      later: { title: 'Clocks', youtubeId: null },
    });
    await fixture.whenStable();
    await new Promise((resolve) => setTimeout(resolve));
    await fixture.whenStable();
    return fixture.nativeElement as HTMLElement;
  }

  it('mounts a player for uploaded videos and a note for the rest', async () => {
    const el = await renderWith('::video[ready]\n\n::video[later]\n\n::video[missing]');
    const slots = [...el.querySelectorAll('.video-slot')];

    expect(slots[0].querySelector('button')?.textContent).toContain('Play animation');
    expect(slots[0].querySelector('figcaption')?.textContent).toContain('Soft lookup');
    expect(slots[1].textContent).toContain('Animation coming soon');
    expect(slots[2].textContent).toBe('(Video “missing” is not available.)');
  });

  it('navigates in-app for links to our own pages', async () => {
    const el = await renderWith('See [Softmax](/concepts/softmax) or [arXiv](https://arxiv.org).');
    const router = TestBed.inject(Router);
    const navigate = vi.spyOn(router, 'navigateByUrl').mockResolvedValue(true);
    const [internal, external] = el.querySelectorAll('a');

    const click = (a: Element) =>
      a.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, button: 0 }));
    expect(click(internal)).toBe(false); // default prevented
    expect(navigate).toHaveBeenCalledWith('/concepts/softmax');
    navigate.mockClear();
    click(external);
    expect(navigate).not.toHaveBeenCalled();
  });

  it('opens a figure image full size in a new tab', async () => {
    const el = await renderWith('![A V neuron](/paper-media/clean/v.webp)');
    const open = vi.spyOn(window, 'open').mockReturnValue(null);

    el.querySelector('img')?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    expect(open).toHaveBeenCalledWith(
      expect.stringContaining('/paper-media/clean/v.webp'),
      '_blank',
      'noopener',
    );
  });
});
