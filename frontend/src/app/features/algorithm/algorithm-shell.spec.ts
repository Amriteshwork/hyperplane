import { TestBed } from '@angular/core/testing';
import { provideRouter, withComponentInputBinding } from '@angular/router';
import { RouterTestingHarness } from '@angular/router/testing';

import { AlgorithmOut } from '../../core/api/models/algorithm-out';
import { CodeOut } from '../../core/api/models/code-out';
import { provideApiTesting, respond, settle } from '../../testing/api-testing';
import { ALGORITHM_ROUTES } from './algorithm.routes';

const algorithm: AlgorithmOut = {
  id: 'linear_regression',
  name: 'Linear Regression',
  summary: 'Fit a line.',
  family: { id: 'linear-models', title: 'Linear Models', summary: '', algorithms: [] },
  task_types: ['regression'],
  difficulty: 1,
  overview: '## Intuition\n\nDraw the **best** line.',
  concepts: [{ id: 'normal-equation', title: 'Normal equation' }],
  resources: [],
  params_schema: {},
  code_variants: [
    { id: 'train', title: 'Training script', filename: 'train.py' },
    { id: 'scratch', title: 'Under the hood (NumPy)', filename: 'scratch.py' },
  ],
};

const code: CodeOut = {
  variant: 'train',
  title: 'Training script',
  filename: 'train.py',
  code: 'import joblib\nmodel.fit(X, y)\n',
  source: 'example',
  annotations: [
    {
      anchor: 'fit',
      start_line: 2,
      end_line: 2,
      short: 'Fit it.',
      deep: 'Deep.',
      concepts: [],
    },
  ],
};

const API = '/api/v1/algorithms/linear_regression';

async function open(path: string, body: AlgorithmOut = algorithm) {
  TestBed.configureTestingModule({
    providers: [
      provideRouter(
        [{ path: 'learn/algorithms/:algorithmId', children: ALGORITHM_ROUTES }],
        withComponentInputBinding(),
      ),
      provideApiTesting(),
    ],
  });
  const harness = await RouterTestingHarness.create();
  const navigation = harness.navigateByUrl(path);
  await respond(API, body);
  await navigation;
  await settle();
  return harness;
}

describe('Algorithm page', () => {
  it('shows the header, tab navigation and the overview by default', async () => {
    const harness = await open('/learn/algorithms/linear_regression');
    const el = harness.routeNativeElement as HTMLElement;

    expect(el.querySelector('h1')?.textContent).toContain('Linear Regression');
    const tabs = [...el.querySelectorAll('nav[aria-label="Algorithm sections"] a')];
    expect(tabs.map((a) => a.textContent?.trim())).toEqual([
      'Overview',
      'Concepts',
      'Resources',
      'Code',
    ]);
    expect(tabs[0].getAttribute('aria-current')).toBe('page');
    expect(el.querySelector('strong')?.textContent).toBe('best');
    expect(el.querySelector('[data-testid="try-it"]')?.getAttribute('href')).toBe(
      '/lab?algorithm=linear_regression',
    );
  });

  it('lists research papers above the overview, and only papers', async () => {
    const resource = {
      title: 'A survey',
      url: 'https://arxiv.org/abs/1609.04747',
      author: 'Ruder',
      level: 'intermediate',
      duration_min: null,
      why: 'Standard survey.',
    } as const;
    const harness = await open('/learn/algorithms/linear_regression', {
      ...algorithm,
      resources: [
        { ...resource, type: 'blog', title: 'A blog', url: 'https://example.org' },
        { ...resource, type: 'paper' },
      ],
    });
    const el = harness.routeNativeElement as HTMLElement;
    const papers = el.querySelector('section[aria-labelledby="papers-heading"]');

    expect([...(papers?.querySelectorAll('h3') ?? [])].map((h) => h.textContent?.trim())).toEqual([
      'A survey (opens in a new tab)',
    ]);
    expect(papers?.compareDocumentPosition(el.querySelector('strong')!)).toBe(
      Node.DOCUMENT_POSITION_FOLLOWING,
    );
  });

  it('has no papers section when there are no papers', async () => {
    const harness = await open('/learn/algorithms/linear_regression');
    expect(harness.routeNativeElement?.textContent).not.toContain('Research papers');
  });

  it('shows an empty state when there are no curated resources', async () => {
    const harness = await open('/learn/algorithms/linear_regression/resources');
    expect(harness.routeNativeElement?.textContent).toContain('Curated resources coming soon');
  });

  it('renders the explained code for a variant', async () => {
    const harness = await open('/learn/algorithms/linear_regression/code');
    await respond(`${API}/code/train`, code);
    const el = harness.routeNativeElement as HTMLElement;

    expect(el.querySelector('[data-anchor="fit"]')?.textContent).toContain('model.fit(X, y)');
    expect(el.textContent).toContain('example settings');
    const variants = el.querySelectorAll('nav[aria-label="Code variants"] a');
    expect(variants[1].getAttribute('href')).toBe(
      '/learn/algorithms/linear_regression/code/scratch',
    );
  });

  it('says so when the algorithm does not exist', async () => {
    TestBed.configureTestingModule({
      providers: [
        provideRouter(
          [{ path: 'learn/algorithms/:algorithmId', children: ALGORITHM_ROUTES }],
          withComponentInputBinding(),
        ),
        provideApiTesting(),
      ],
    });
    const harness = await RouterTestingHarness.create();
    const navigation = harness.navigateByUrl('/learn/algorithms/nope');
    await respond(
      '/api/v1/algorithms/nope',
      { detail: 'Unknown' },
      { status: 404, statusText: 'Not Found' },
    );
    await navigation;
    expect(harness.routeNativeElement?.textContent).toContain('Algorithm not found');
  });
});
