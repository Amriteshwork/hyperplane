import { Location } from '@angular/common';
import { HttpTestingController } from '@angular/common/http/testing';
import { Component, input } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { provideRouter, withComponentInputBinding } from '@angular/router';
import { RouterTestingHarness } from '@angular/router/testing';

import { RunOut } from '../../core/api/models/run-out';
import { TrainingOptionsOut } from '../../core/api/models/training-options-out';
import { EVENT_SOURCE_FACTORY } from '../../core/http/event-stream';
import { Chart } from '../../shared/ui/chart/chart';
import { provideApiTesting, respond, settle } from '../../testing/api-testing';
import { LAB_ROUTES } from './lab.routes';
import { RunResults } from './run-results';

/** jsdom can't draw on a canvas; plot options have their own unit test. */
@Component({
  selector: 'app-chart',
  template: '<figure [attr.aria-label]="label()"><ng-content /></figure>',
})
class StubChart {
  readonly options = input<object>();
  readonly label = input('');
  readonly caption = input<string>();
  readonly dark = input(false);
}

/** Stands in for the browser's EventSource; tests push messages with `emit`. */
class FakeEventSource {
  static last: FakeEventSource | null = null;
  onmessage: ((message: MessageEvent<string>) => void) | null = null;
  onerror: (() => void) | null = null;
  closed = false;

  constructor(readonly url: string) {
    FakeEventSource.last = this;
  }

  emit(data: object): void {
    this.onmessage?.(new MessageEvent('message', { data: JSON.stringify(data) }));
  }

  close(): void {
    this.closed = true;
  }
}

const dataset = {
  id: 'd1',
  name: 'Diabetes',
  source: 'sample',
  sample_id: 'diabetes',
  format: 'csv',
  size_bytes: 1,
  n_rows: 442,
  n_columns: 3,
  status: 'ready',
  error: null,
  created_at: '2026-01-01T00:00:00Z',
  expires_at: '2026-01-08T00:00:00Z',
};
const options: TrainingOptionsOut = {
  targets: [
    {
      column: 'progression',
      task_type: 'regression',
      reason: '214 distinct numeric values → regression.',
      algorithms: [{ id: 'linear_regression', name: 'Linear Regression' }],
    },
    { column: 'sex', task_type: 'binary', reason: '2 distinct values → binary.', algorithms: [] },
  ],
  features: [
    { column: 'bmi', semantic_type: 'numeric', role: 'numeric' },
    { column: 'sex', semantic_type: 'boolean', role: 'categorical' },
    { column: 'visit', semantic_type: 'datetime', role: null },
    { column: 'progression', semantic_type: 'numeric', role: 'numeric' },
  ],
};
const queued: RunOut = {
  id: 'r1',
  status: 'queued',
  algorithm_id: 'linear_regression',
  algorithm_version: 'scikit-learn 1.9.1',
  params: { fit_intercept: true, positive: false },
  created_at: '2026-01-01T00:00:00Z',
  started_at: null,
  finished_at: null,
  error: null,
  experiment: {
    id: 'e1',
    dataset_id: 'd1',
    target: 'progression',
    task_type: 'regression',
    numeric_features: ['bmi'],
    categorical_features: ['sex'],
    test_size: 0.2,
    seed: 42,
  },
  report: null,
};
const succeeded: RunOut = {
  ...queued,
  status: 'succeeded',
  report: {
    n_train: 353,
    n_test: 89,
    evaluation: {
      metrics: [
        {
          id: 'r2',
          label: 'R²',
          value: 0.45261,
          description: 'Share of variance explained.',
          higher_is_better: true,
        },
      ],
      plots: [
        {
          kind: 'scatter',
          id: 'predicted-vs-actual',
          title: 'Predicted vs actual',
          description: 'Each dot is a test row.',
          x_label: 'Actual',
          y_label: 'Predicted',
          points: [
            [1, 1.5],
            [2, 2],
          ],
          reference: 'identity',
        },
      ],
    },
    coefficients: [{ feature: 'numeric__bmi', value: 25.1234567 }],
  },
};

async function open(url: string) {
  TestBed.configureTestingModule({
    providers: [
      provideRouter([{ path: 'lab', children: LAB_ROUTES }], withComponentInputBinding()),
      provideApiTesting(),
      { provide: EVENT_SOURCE_FACTORY, useValue: (u: string) => new FakeEventSource(u) },
    ],
  });
  TestBed.overrideComponent(RunResults, {
    remove: { imports: [Chart] },
    add: { imports: [StubChart] },
  });
  const harness = await RouterTestingHarness.create();
  const navigation = harness.navigateByUrl(url);
  return { harness, navigation };
}

describe('Lab page', () => {
  beforeEach(() => (FakeEventSource.last = null));

  it('creates a dataset from a sample and moves on once it is profiled', async () => {
    const { harness, navigation } = await open('/lab');
    await respond('/api/v1/samples?limit=100', {
      items: [
        {
          id: 'diabetes',
          name: 'Diabetes',
          description: 'Disease progression.',
          task: 'regression',
          target: 'progression',
          rows: 442,
          columns: 11,
          status: 'available',
          licence: '',
          source: '',
          url: null,
          citation: null,
        },
      ],
      total: 1,
      limit: 100,
      offset: 0,
    });
    await navigation;
    const el = harness.routeNativeElement as HTMLElement;

    (el.querySelector('section[aria-labelledby="samples-heading"] button') as HTMLElement).click();
    await respond('/api/v1/samples/diabetes/datasets', { ...dataset, status: 'profiling' });
    await respond('/api/v1/datasets/d1', { ...dataset, status: 'profiling' });
    expect(TestBed.inject(Location).path()).toBe('/lab?dataset=d1');
    expect(el.textContent).toContain('Profiling');
  });

  it('defaults the target and usable features, then trains and shows results', async () => {
    const { harness, navigation } = await open('/lab?dataset=d1');
    await respond('/api/v1/datasets/d1', dataset);
    await respond('/api/v1/datasets/d1/training-options', options);
    await navigation;
    const el = harness.routeNativeElement as HTMLElement;

    const target = el.querySelector<HTMLSelectElement>('#lab-target')!;
    expect(target.value).toBe('progression');
    const boxes = [...el.querySelectorAll<HTMLInputElement>('fieldset input[type="checkbox"]')];
    expect(boxes.map((b) => [b.checked, b.disabled])).toEqual([
      [true, false], // bmi
      [true, false], // sex
      [false, true], // visit: datetime can't be a feature yet
    ]);

    [...el.querySelectorAll('button')].find((b) => b.textContent?.includes('Next'))!.click();
    await respond('/api/v1/algorithms/linear_regression', {
      params_schema: {
        type: 'object',
        properties: {
          fit_intercept: {
            type: 'boolean',
            default: true,
            title: 'Fit intercept',
            level: 'beginner',
          },
        },
      },
    });
    [...el.querySelectorAll('button')].find((b) => b.textContent?.includes('Train model'))!.click();
    await settle();
    const [create] = TestBed.inject(HttpTestingController).match('/api/v1/experiments');
    expect(create.request.body).toEqual({
      dataset_id: 'd1',
      target: 'progression',
      features: ['bmi', 'sex'],
      algorithm_id: 'linear_regression',
      params: {},
      split: { test_size: 0.2 },
      seed: 42,
    });
    create.flush(queued);
    await respond('/api/v1/runs/r1', queued);
    expect(TestBed.inject(Location).path()).toBe('/lab?dataset=d1&run=r1');

    const source = FakeEventSource.last!;
    expect(source.url).toBe('/api/v1/runs/r1/events');
    source.emit({ type: 'status', status: 'running' });
    source.emit({ type: 'progress', step: 1, total: 3, message: 'Fitting the model' });
    await settle();
    expect(el.textContent).toContain('Fitting the model');

    source.emit({ type: 'done', status: 'succeeded' });
    expect(source.closed).toBe(true);
    await respond('/api/v1/runs/r1', succeeded);
    expect(el.querySelector('[data-testid="run-status"]')?.textContent?.trim()).toBe('succeeded');
    expect(el.textContent).toContain('0.4526');
    expect(el.querySelector('figure')?.getAttribute('aria-label')).toBe('Each dot is a test row.');
    expect(el.textContent).toContain('bmi');
    expect(el.textContent).toContain('25.12');
  });

  it('shows why a run failed', async () => {
    const { harness, navigation } = await open('/lab?dataset=d1&run=r1');
    await respond('/api/v1/datasets/d1', dataset);
    await respond('/api/v1/runs/r1', {
      ...queued,
      status: 'failed',
      error: 'Training failed: boom',
    });
    await navigation;
    expect(harness.routeNativeElement?.textContent).toContain('Training failed: boom');
    expect(FakeEventSource.last).toBeNull(); // a finished run isn't followed
  });
});
