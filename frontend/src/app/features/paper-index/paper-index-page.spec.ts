import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';

import { provideApiTesting, respond, settle } from '../../testing/api-testing';
import { PaperIndexPage } from './paper-index-page';

describe('PaperIndexPage', () => {
  it('lists each walkthrough with its source', async () => {
    TestBed.configureTestingModule({ providers: [provideRouter([]), provideApiTesting()] });
    const fixture = TestBed.createComponent(PaperIndexPage);
    fixture.detectChanges();
    await respond('/api/v1/papers?limit=100', {
      items: [
        {
          id: 'attention-is-all-you-need',
          title: 'Attention Is All You Need, explained',
          summary: 'The transformer paper.',
          source: {
            type: 'paper',
            title: 'Attention',
            url: 'https://arxiv.org',
            author: 'Vaswani et al.',
          },
          section_count: 12,
        },
      ],
      total: 1,
      limit: 100,
      offset: 0,
    });
    await settle();
    const el = fixture.nativeElement as HTMLElement;

    const card = el.querySelector('a[href="/papers/attention-is-all-you-need"]');
    expect(card?.textContent?.replace(/\s+/g, ' ')).toContain('Paper · Vaswani et al.');
    expect(card?.textContent).toContain('12 sections');
  });
});
