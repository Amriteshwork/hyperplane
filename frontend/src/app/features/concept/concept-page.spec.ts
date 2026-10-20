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
  learning_path: [
    { id: 'loss-function', title: 'Loss function' },
    { id: 'mean-squared-error', title: 'Mean squared error (MSE)' },
  ],
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
  it('shows the body, learning path, related algorithms and resource empty state', async () => {
    const el = await open(concept);

    expect(el.querySelector('h1')?.textContent).toContain('Normal equation');
    const path = el.querySelector('nav[aria-labelledby="learning-path"]')!;
    expect([...path.querySelectorAll('a')].map((a) => a.getAttribute('href'))).toEqual([
      '/concepts/loss-function',
      '/concepts/mean-squared-error',
    ]);
    expect(path.querySelector('[aria-current="page"]')?.textContent).toContain('Normal equation');
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
          uses: [
            {
              concept: { id: 'mean-squared-error', title: 'Mean squared error (MSE)' },
              filename: 'mean_squared_error_numpy.py',
              names: ['mse'],
            },
          ],
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
          uses: [],
        },
      ],
    });
    const tabs = [...el.querySelectorAll<HTMLButtonElement>('[role="tab"]')];
    const panel = () => el.querySelector('[role="tabpanel"]')!;

    expect(tabs.map((t) => t.getAttribute('aria-selected'))).toEqual(['true', 'false']);
    expect(panel().textContent).toContain('normal_equation_numpy.py');
    const uses = panel().querySelector('[data-testid="code-uses"]')!;
    expect(uses.textContent).toContain('mse');
    expect(uses.textContent).toContain('mean_squared_error_numpy.py');
    expect(uses.querySelector('a')?.getAttribute('href')).toBe('/concepts/mean-squared-error');
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
    expect(panel().querySelector('[data-testid="code-uses"]')).toBeNull(); // standalone
  });

  describe('download with dependencies', () => {
    const BUNDLE = '/api/v1/concepts/normal-equation/code/numpy/bundle';
    const withUses: ConceptOut = {
      ...concept,
      code: [
        {
          library: 'numpy',
          title: 'NumPy',
          filename: 'normal_equation_numpy.py',
          code: 'from mean_squared_error_numpy import mse\n',
          annotations: [],
          uses: [
            {
              concept: { id: 'mean-squared-error', title: 'Mean squared error (MSE)' },
              filename: 'mean_squared_error_numpy.py',
              names: ['mse'],
            },
          ],
        },
      ],
    };
    const button = (el: HTMLElement) =>
      [...el.querySelectorAll('button')].find((b) => b.textContent?.includes('dependencies'))!;

    afterEach(() => vi.restoreAllMocks());

    it('saves the zip of this file and everything it imports', async () => {
      const createObjectURL = vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:zip');
      vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined);
      const saved: string[] = [];
      vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (
        this: HTMLAnchorElement,
      ) {
        saved.push(this.download);
      });
      const download = button(await open(withUses));

      download.click();
      await settle();
      expect(download.disabled).toBe(true);
      expect(download.textContent).toContain('Preparing');
      await respond(BUNDLE, new Blob(['PK']));

      expect(saved).toEqual(['normal_equation_numpy.zip']);
      expect(createObjectURL).toHaveBeenCalledOnce();
      expect(download.disabled).toBe(false);
    });

    it('says so when the zip cannot be made', async () => {
      const el = await open(withUses);

      button(el).click();
      await respond(BUNDLE, new Blob(['']), { status: 500, statusText: 'Server Error' });

      expect(el.querySelector('[role="alert"]')).not.toBeNull();
      expect(button(el).disabled).toBe(false);
    });
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
          note: 'Read section 4 for momentum.',
        },
      ],
    });
    const papers = el.querySelector('section[aria-labelledby="papers"]');
    const resources = el.querySelector('section[aria-labelledby="resources"]');

    expect(papers?.textContent).toContain('A survey');
    expect(papers?.querySelector('[data-testid="resource-note"]')?.textContent).toContain(
      'Read section 4 for momentum.',
    );
    expect(resources?.textContent).not.toContain('A survey');
  });
});
