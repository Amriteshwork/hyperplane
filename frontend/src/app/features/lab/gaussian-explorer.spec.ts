import { Component, input } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';

import { GaussianReport } from '../../core/api/models/gaussian-report';
import { Chart } from '../../shared/ui/chart/chart';
import { GaussianExplorer } from './gaussian-explorer';

/** jsdom can't draw on a canvas; `bellOptions` has its own unit test. */
@Component({ selector: 'app-chart', template: '<ng-content />' })
class StubChart {
  readonly options = input<object>();
  readonly label = input('');
  readonly caption = input<string>();
  readonly dark = input(false);
}

const report: GaussianReport = {
  classes: ['ant', 'bee'],
  priors: [0.25, 0.75],
  features: [
    { feature: 'numeric__bmi', means: [20, 30], stds: [1, 2], separation: 3 },
    { feature: 'categorical__sex_M', means: [0, 1], stds: [0.1, 0.1], separation: 2 },
  ],
  total_features: 6,
  mean_confidence: 0.99,
  accuracy: 0.9,
  correlated: { a: 'numeric__bmi', b: 'numeric__age', correlation: 0.87 },
  examples: [
    {
      kind: 'miss',
      predicted: 'bee',
      actual: 'ant',
      rival: 'ant',
      confidence: 0.75,
      prior: { feature: 'Prior', value: '', predicted: -0.5, rival: -1.5 },
      evidence: [{ feature: 'numeric__bmi', value: '25', predicted: -2, rival: -3 }],
      others: { feature: '5 other columns', value: '', predicted: -10, rival: -10.5 },
    },
    {
      kind: 'hit',
      predicted: 'ant',
      actual: 'ant',
      rival: 'bee',
      confidence: 0.98,
      prior: { feature: 'Prior', value: '', predicted: -1.5, rival: -0.5 },
      evidence: [{ feature: 'categorical__sex_M', value: '0', predicted: 1.4, rival: -48 }],
      others: null,
    },
  ],
};

function render(value = report) {
  TestBed.configureTestingModule({ providers: [provideRouter([])] });
  TestBed.overrideComponent(GaussianExplorer, {
    remove: { imports: [Chart] },
    add: { imports: [StubChart] },
  });
  const fixture = TestBed.createComponent(GaussianExplorer);
  fixture.componentRef.setInput('report', value);
  fixture.detectChanges();
  return fixture;
}

const cells = (row: Element) =>
  [...row.querySelectorAll('td, th')].map((c) => c.textContent?.trim());

describe('GaussianExplorer', () => {
  it('adds up the biggest miss, in rows that total to each class score', () => {
    const element = render().nativeElement as HTMLElement;
    const rows = [...element.querySelectorAll('tbody tr')];
    expect(cells(rows[0])).toEqual(['Prior', '', '-0.5', '-1.5']);
    expect(cells(rows[1])).toEqual(['bmi', '25', '-2', '-3']);
    expect(cells(rows[2])).toEqual(['5 other columns', '', '-10', '-10.5']);
    expect(cells(rows[3])).toEqual(['Total', '', '-12.5', '-15']);
    const headers = [...element.querySelectorAll('th')].map((h) => h.textContent?.trim());
    expect(headers).toContain('bee (predicted)');
    expect(headers).toContain('ant (rival)');
  });

  it('switches test row, leaving out the other-columns row when nothing was grouped', async () => {
    const fixture = render();
    const element = fixture.nativeElement as HTMLElement;
    const select = element.querySelector('#gaussian-example') as HTMLSelectElement;
    expect(select.options[0].textContent).toContain('Big miss: predicted bee');
    select.value = '1';
    select.dispatchEvent(new Event('change'));
    await fixture.whenStable();
    const rows = [...element.querySelectorAll('tbody tr')];
    expect(rows).toHaveLength(3);
    expect(cells(rows[1])).toEqual(['sex_M', '0', '1.4', '-48']);
  });

  it('describes the chosen feature for screen readers and lists the separators', async () => {
    const fixture = render();
    const element = fixture.nativeElement as HTMLElement;
    const chart = () => fixture.debugElement.query((d) => d.componentInstance instanceof StubChart);
    expect((chart().componentInstance as StubChart).label()).toContain('ant: mean 20, spread 1');
    const select = element.querySelector('#gaussian-feature') as HTMLSelectElement;
    select.value = '1';
    select.dispatchEvent(new Event('change'));
    await fixture.whenStable();
    expect((chart().componentInstance as StubChart).label()).toContain('Bell curves of sex_M');
    expect(element.textContent).toContain('the top 2 of 6');
  });

  it('says when the model is more sure than it has earned, and why that may be', () => {
    const element = render().nativeElement as HTMLElement;
    expect(element.textContent).toContain('more sure of itself');
    expect(element.textContent).toContain('bmi and age');
    expect(element.querySelector('a')?.getAttribute('href')).toBe(
      '/concepts/conditional-independence',
    );
    expect(element.textContent).toContain('ant 25.0%, bee 75.0%');
  });

  it('leaves out what does not apply', () => {
    const element = render({ ...report, mean_confidence: 0.9, correlated: null, total_features: 2 })
      .nativeElement as HTMLElement;
    expect(element.textContent).toContain('in line with its accuracy');
    expect(element.textContent).not.toContain('Most correlated');
    expect(element.textContent).not.toContain('the top 2');
  });
});
