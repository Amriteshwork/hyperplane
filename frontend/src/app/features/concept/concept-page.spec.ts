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
  code: [],
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
    expect(el.querySelector('#from-scratch')).toBeNull(); // no code, no section
  });

  it('shows the from-scratch code one library at a time, with its explanations', async () => {
    const el = await open({
      ...concept,
      code: [
        {
          library: 'numpy',
          title: 'NumPy',
          filename: 'normal_equation_numpy.py',
          code: 'import numpy as np\nbeta = np.linalg.solve(A, b)\n',
          annotations: [
            {
              anchor: 'solve',
              start_line: 2,
              end_line: 2,
              short: 'Solve the normal equation.',
              deep: 'In depth.',
              concepts: [{ id: 'mean-squared-error', title: 'Mean squared error (MSE)' }],
            },
          ],
        },
        {
          library: 'pytorch',
          title: 'PyTorch',
          filename: 'normal_equation_pytorch.py',
          code: 'import torch\n',
          annotations: [],
        },
      ],
    });
    const tabs = [...el.querySelectorAll<HTMLButtonElement>('[role="tab"]')];
    const panel = () => el.querySelector('[role="tabpanel"]')!;

    expect(tabs.map((t) => t.getAttribute('aria-selected'))).toEqual(['true', 'false']);
    expect(panel().textContent).toContain('normal_equation_numpy.py');
    panel().querySelector<HTMLButtonElement>('[data-anchor="solve"]')!.click();
    await settle();
    expect(panel().querySelector('aside')?.textContent).toContain('Solve the normal equation.');

    tabs[0].dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));
    await settle();
    expect(tabs[1].getAttribute('aria-selected')).toBe('true');
    expect(document.activeElement).toBe(tabs[1]);
    expect(panel().textContent).toContain('normal_equation_pytorch.py');
    expect(panel().getAttribute('aria-labelledby')).toBe('concept-code-tab-pytorch');
    expect(panel().querySelector('aside')).toBeNull(); // nothing to explain beside it
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
