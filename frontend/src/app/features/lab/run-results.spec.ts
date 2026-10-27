import { Component, input } from '@angular/core';
import { TestBed } from '@angular/core/testing';

import { TrainingReport } from '../../core/api/models/training-report';
import { Chart } from '../../shared/ui/chart/chart';
import { featureLabel } from './feature-label';
import { RunResults } from './run-results';

/** jsdom can't draw on a canvas; plot options have their own unit test. */
@Component({ selector: 'app-chart', template: '<ng-content />' })
class StubChart {
  readonly options = input<object>();
  readonly label = input('');
  readonly caption = input<string>();
  readonly dark = input(false);
  readonly height = input('');
}

function render(report: TrainingReport): HTMLElement {
  TestBed.overrideComponent(RunResults, {
    remove: { imports: [Chart] },
    add: { imports: [StubChart] },
  });
  const fixture = TestBed.createComponent(RunResults);
  fixture.componentRef.setInput('report', report);
  fixture.detectChanges();
  return fixture.nativeElement as HTMLElement;
}

const evaluation = { metrics: [], plots: [] };

describe('RunResults coefficients', () => {
  it('adds a Class column when each class has its own weights', () => {
    const element = render({
      n_train: 8,
      n_test: 2,
      evaluation,
      coefficients: [{ feature: 'numeric__bmi', value: 0.5, target_class: 'malignant' }],
    });
    const headers = [...element.querySelectorAll('th')].map((th) => th.textContent?.trim());
    expect(headers).toEqual(['Class', 'Feature', 'Coefficient']);
    expect(element.querySelector('tbody tr')?.textContent).toContain('malignant');
    expect(element.textContent).toContain('log-odds of that class');
  });

  it('keeps two columns for a regression', () => {
    const element = render({
      n_train: 8,
      n_test: 2,
      evaluation,
      coefficients: [{ feature: 'numeric__bmi', value: 0.5 }],
    });
    const headers = [...element.querySelectorAll('th')].map((th) => th.textContent?.trim());
    expect(headers).toEqual(['Feature', 'Coefficient']);
    expect(element.textContent).toContain('push the prediction up');
  });
});

describe('RunResults networks', () => {
  it('shows the trained network and what its first layer listens to', () => {
    const element = render({
      n_train: 8,
      n_test: 2,
      evaluation,
      network: {
        layers: [
          { kind: 'input', units: 2, activation: null, shown: ['numeric__a', 'numeric__b'] },
          { kind: 'hidden', units: 4, activation: 'relu', shown: ['neuron 1'] },
          { kind: 'output', units: 1, activation: 'identity', shown: ['prediction'] },
        ],
        parameters: 17,
        weights: [[[1], [2]], [[3]]],
      },
      feature_importance: [
        { feature: 'numeric__b', value: 2 },
        { feature: 'numeric__a', value: 1 },
      ],
    });
    const headings = [...element.querySelectorAll('h2')].map((h) => h.textContent?.trim());
    expect(headings).toContain('The trained network');
    expect(headings).toContain('What the first layer listens to');
    expect(element.textContent).toContain('2 → 4 relu → 1: 17 weights and biases');
    expect(element.querySelector('app-network-diagram svg')).not.toBeNull();
  });

  it('leaves both sections out for other models', () => {
    const headings = [...render({ n_train: 8, n_test: 2, evaluation }).querySelectorAll('h2')].map(
      (h) => h.textContent?.trim(),
    );
    expect(headings).not.toContain('The trained network');
    expect(headings).not.toContain('What the first layer listens to');
  });
});

describe('featureLabel', () => {
  it('drops the preprocessing prefix', () => {
    expect(featureLabel('numeric__bmi')).toBe('bmi');
    expect(featureLabel('categorical__island_Dream')).toBe('island_Dream');
  });

  it('keeps powers and joins the factors of a product term', () => {
    expect(featureLabel('numeric__age^2')).toBe('age^2');
    expect(featureLabel('numeric__age numeric__bmi')).toBe('age × bmi');
    expect(featureLabel('numeric__a^2 categorical__sex_M')).toBe('a^2 × sex_M');
  });

  it('keeps spaces inside a column name', () => {
    expect(featureLabel('numeric__body mass')).toBe('body mass');
    expect(featureLabel('numeric__body mass numeric__flipper length')).toBe(
      'body mass × flipper length',
    );
  });

  it('leaves names without a prefix unchanged', () => {
    expect(featureLabel('x0 x1')).toBe('x0 x1');
  });
});
