import { TestBed } from '@angular/core/testing';
import { provideRouter, withComponentInputBinding } from '@angular/router';
import { RouterTestingHarness } from '@angular/router/testing';

import { PaperOut } from '../../core/api/models/paper-out';
import { FIGURES } from '../../shared/figures/figure-registry';
import { provideApiTesting, respond, settle } from '../../testing/api-testing';
import { PAPER_ROUTES } from './paper.routes';

const paper: PaperOut = {
  id: 'attention-is-all-you-need',
  title: 'Attention Is All You Need, explained',
  summary: 'The transformer paper, section by section.',
  source: {
    type: 'paper',
    title: 'Attention Is All You Need',
    url: 'https://arxiv.org/abs/1706.03762',
    author: 'Vaswani et al. (2017)',
  },
  intro: 'Why it **matters**.',
  sections: [
    { id: '1-introduction', title: '1 · Introduction', body: 'RNNs are slow.\n\n::video[chain]\n' },
    {
      id: '3-5-positions',
      title: '3.5 · Positions',
      body: 'Clocks, see [it](/concepts/positional-encoding).\n',
    },
  ],
  videos: [{ id: 'chain', title: 'One word at a time', youtube_id: null }],
  concepts: [
    { id: 'softmax', title: 'Softmax', summary: 'Scores to shares.', algorithms: [] },
    { id: 'self-attention', title: 'Self-attention', summary: 'Look around.', algorithms: [] },
  ],
  track: {
    id: 'transformers',
    title: 'Transformers, from scratch',
    summary: 's',
    goal: { id: 'transformer', title: 'Transformer' },
    concept_count: 32,
  },
  resources: [],
};

async function open(status = 200) {
  TestBed.configureTestingModule({
    providers: [
      provideRouter(
        [{ path: 'papers/:paperId', children: PAPER_ROUTES }],
        withComponentInputBinding(),
      ),
      provideApiTesting(),
      { provide: FIGURES, useValue: {} },
    ],
  });
  const harness = await RouterTestingHarness.create();
  const navigation = harness.navigateByUrl('/papers/attention-is-all-you-need');
  await respond(
    '/api/v1/papers/attention-is-all-you-need',
    status === 200 ? paper : { title: 'nope' },
    { status, statusText: status === 200 ? 'OK' : 'Not Found' },
  );
  await navigation;
  await settle();
  return harness.routeNativeElement as HTMLElement;
}

describe('PaperPage', () => {
  it('shows the source, what to read first, an outline and every section', async () => {
    const el = await open();
    const text = el.textContent?.replace(/\s+/g, ' ');

    expect(el.querySelector('h1')?.textContent).toContain('Attention Is All You Need, explained');
    expect(text).toContain('Paper walkthrough');
    const original = [...el.querySelectorAll('a')].find((a) => a.href.includes('arxiv'));
    expect(original?.getAttribute('target')).toBe('_blank');
    expect(text).toContain('Vaswani et al. (2017)');

    const before = el.querySelector('aside[aria-labelledby="before-you-read"]');
    expect([...(before?.querySelectorAll('a') ?? [])].map((a) => a.getAttribute('href'))).toEqual([
      '/concepts/softmax',
      '/concepts/self-attention',
      '/tracks/transformers',
    ]);

    const outline = [...el.querySelectorAll('nav[aria-labelledby="outline"] a')];
    expect(outline.map((a) => a.getAttribute('href'))).toEqual([
      '/papers/attention-is-all-you-need#1-introduction',
      '/papers/attention-is-all-you-need#3-5-positions',
    ]);
    const sections = el.querySelectorAll('section[id]');
    expect([...sections].map((s) => s.querySelector('h2')?.textContent?.trim())).toEqual([
      '1 · Introduction',
      '3.5 · Positions',
    ]);
    expect(el.querySelector('strong')?.textContent).toBe('matters');
    expect(sections[0].querySelector('.video-slot')?.textContent).toContain(
      'Animation coming soon',
    );
  });

  it('says so when the walkthrough does not exist', async () => {
    const el = await open(404);
    expect(el.textContent).toContain('Walkthrough not found');
  });
});
