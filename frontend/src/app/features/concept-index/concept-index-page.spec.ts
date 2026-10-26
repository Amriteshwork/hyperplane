import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';

import { ConceptAreaOut } from '../../core/api/models/concept-area-out';
import { provideApiTesting, respond, settle } from '../../testing/api-testing';
import { ConceptIndexPage } from './concept-index-page';

const areas: ConceptAreaOut[] = [
  {
    id: 'foundations',
    title: 'Maths & NumPy foundations',
    concepts: [
      { id: 'vectors', title: 'Vectors', summary: 'Lists of numbers.', algorithms: [] },
      {
        id: 'derivatives',
        title: 'Derivatives and gradients',
        summary: 'Slopes that point uphill.',
        algorithms: [],
      },
    ],
  },
  {
    id: 'optimisation',
    title: 'Loss & optimisation',
    concepts: [
      {
        id: 'gradient-descent',
        title: 'Gradient descent',
        summary: 'Step downhill along the gradient.',
        algorithms: [{ id: 'linear_regression', name: 'Linear Regression' }],
      },
    ],
  },
];

async function open() {
  TestBed.configureTestingModule({ providers: [provideRouter([]), provideApiTesting()] });
  const fixture = TestBed.createComponent(ConceptIndexPage);
  await respond('/api/v1/concepts?limit=100', { items: areas, total: 2, limit: 100, offset: 0 });
  await settle();
  const el = fixture.nativeElement as HTMLElement;
  const search = async (query: string) => {
    const input = el.querySelector<HTMLInputElement>('#concept-search')!;
    input.value = query;
    input.dispatchEvent(new Event('input'));
    await settle();
  };
  const links = () =>
    [...el.querySelectorAll('section[aria-labelledby^="area-"] a')].map((a) =>
      a.getAttribute('href'),
    );
  return { el, search, links };
}

describe('ConceptIndexPage', () => {
  it('lists every concept under its area, linking to its page', async () => {
    const { el, links } = await open();

    expect([...el.querySelectorAll('h2')].map((h) => h.textContent?.trim())).toEqual([
      'Maths & NumPy foundations',
      'Loss & optimisation',
    ]);
    expect(links()).toEqual([
      '/concepts/vectors',
      '/concepts/derivatives',
      '/concepts/gradient-descent',
    ]);
    expect(el.textContent?.replace(/\s+/g, ' ')).toContain('Used by Linear Regression');
  });

  it('searches titles and summaries, hiding empty areas', async () => {
    const { el, search, links } = await open();

    await search('gradient');
    expect(links()).toEqual(['/concepts/derivatives', '/concepts/gradient-descent']);
    expect(el.querySelector('[role="status"]')?.textContent).toContain('2 of 3 concepts');

    await search('downhill step');
    expect(links()).toEqual(['/concepts/gradient-descent']);
    expect(el.querySelector('#area-foundations')).toBeNull();
  });

  it('says so when nothing matches, and can clear the search', async () => {
    const { el, search, links } = await open();

    await search('transformer');
    expect(el.textContent).toContain('No concepts match');
    [...el.querySelectorAll('button')].find((b) => b.textContent?.includes('Clear'))!.click();
    await settle();
    expect(links()).toHaveLength(3);
  });
});
