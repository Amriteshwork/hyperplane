import { TestBed } from '@angular/core/testing';
import { provideRouter, withComponentInputBinding } from '@angular/router';
import { RouterTestingHarness } from '@angular/router/testing';

import { ConceptOut } from '../../core/api/models/concept-out';
import { provideApiTesting, respond, settle } from '../../testing/api-testing';
import { CONCEPT_ROUTES } from './concept.routes';

const concept: ConceptOut = {
  id: 'normal-equation',
  title: 'Normal equation',
  summary: 'Closed form.',
  body: 'Solve $X^\\top X \\beta = X^\\top y$ directly.',
  prerequisites: [{ id: 'mean-squared-error', title: 'Mean squared error (MSE)' }],
  resources: [],
  algorithms: [{ id: 'linear_regression', name: 'Linear Regression' }],
};

async function open(body: ConceptOut) {
  TestBed.configureTestingModule({
    providers: [
      provideRouter(
        [{ path: 'concepts/:conceptId', children: CONCEPT_ROUTES }],
        withComponentInputBinding(),
      ),
      provideApiTesting(),
    ],
  });
  const harness = await RouterTestingHarness.create();
  const navigation = harness.navigateByUrl('/concepts/normal-equation');
  await respond('/api/v1/concepts/normal-equation', body);
  await navigation;
  await settle();
  return harness.routeNativeElement as HTMLElement;
}

describe('ConceptPage', () => {
  it('shows the body, prerequisites, related algorithms and resource empty state', async () => {
    const el = await open(concept);

    expect(el.querySelector('h1')?.textContent).toContain('Normal equation');
    expect(el.querySelector('a[href="/concepts/mean-squared-error"]')).not.toBeNull();
    expect(el.querySelector('a[href="/learn/algorithms/linear_regression"]')).not.toBeNull();
    expect(el.querySelector('.math')).not.toBeNull();
    expect(el.textContent).toContain('Curated resources coming soon');
    expect(el.querySelector('#papers')).toBeNull();
  });

  it('lists papers in their own section, apart from other resources', async () => {
    const el = await open({
      ...concept,
      resources: [
        {
          type: 'paper',
          title: 'A survey',
          url: 'https://arxiv.org/abs/1609.04747',
          author: null,
          level: 'advanced',
          duration_min: null,
          why: 'Deep dive.',
        },
      ],
    });
    const papers = el.querySelector('section[aria-labelledby="papers"]');
    const resources = el.querySelector('section[aria-labelledby="resources"]');

    expect(papers?.textContent).toContain('A survey');
    expect(resources?.textContent).not.toContain('A survey');
  });
});
