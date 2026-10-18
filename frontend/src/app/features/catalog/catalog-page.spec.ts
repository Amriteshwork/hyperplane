import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';

import { FamilyOut } from '../../core/api/models/family-out';
import { provideApiTesting, respond } from '../../testing/api-testing';
import { CatalogPage } from './catalog-page';

const families: FamilyOut[] = [
  {
    id: 'linear-models',
    title: 'Linear Models',
    summary: 'Weighted sums.',
    algorithms: [
      {
        id: 'linear_regression',
        name: 'Linear Regression',
        available: true,
        summary: 'Fit a line.',
        task_types: ['regression'],
        difficulty: 1,
      },
      {
        id: 'ridge',
        name: 'Ridge',
        available: false,
        summary: null,
        task_types: [],
        difficulty: null,
      },
    ],
  },
];

describe('CatalogPage', () => {
  it('links available algorithms and greys out upcoming ones', async () => {
    TestBed.configureTestingModule({ providers: [provideRouter([]), provideApiTesting()] });
    const fixture = TestBed.createComponent(CatalogPage);
    await respond('/api/v1/families?limit=100', {
      items: families,
      total: 1,
      limit: 100,
      offset: 0,
    });
    const el: HTMLElement = fixture.nativeElement;

    expect(el.querySelector('h2')?.textContent).toContain('Linear Models');
    const available = el.querySelector('[data-testid="algorithm-linear_regression"]');
    expect(available?.tagName).toBe('A');
    expect(available?.getAttribute('href')).toBe('/learn/algorithms/linear_regression');
    expect(available?.textContent).toContain('Regression');

    const upcoming = el.querySelector('[data-testid="algorithm-ridge"]');
    expect(upcoming?.tagName).not.toBe('A');
    expect(upcoming?.querySelector('a')).toBeNull();
    expect(upcoming?.textContent).toContain('Coming soon');
  });

  it('offers a retry when the catalog fails to load', async () => {
    TestBed.configureTestingModule({ providers: [provideRouter([]), provideApiTesting()] });
    const fixture = TestBed.createComponent(CatalogPage);
    await respond('/api/v1/families?limit=100', null, { status: 500, statusText: 'Error' });

    expect(fixture.nativeElement.textContent).toContain('Couldn’t load the catalog');
    expect(fixture.nativeElement.textContent).toContain('Try again');
  });
});
