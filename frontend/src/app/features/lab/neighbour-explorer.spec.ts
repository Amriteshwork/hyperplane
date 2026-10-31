import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';

import { NeighbourReport } from '../../core/api/models/neighbour-report';
import { NeighbourExplorer } from './neighbour-explorer';

const report: NeighbourReport = {
  k: 2,
  stored_rows: 1200,
  columns: ['bmi', 'age'],
  total_columns: 5,
  nearest_distance: 1,
  typical_distance: 4,
  examples: [
    {
      kind: 'miss',
      predicted: 'yes',
      actual: 'no',
      values: ['30', '50'],
      neighbours: [
        { distance: 0.5, share: 0.75, target: 'yes', values: ['31', '49'] },
        { distance: 1.5, share: 0.25, target: 'no', values: ['28', '61'] },
      ],
    },
    {
      kind: 'hit',
      predicted: 'no',
      actual: 'no',
      values: ['20', '22'],
      neighbours: [{ distance: 0.25, share: 1, target: 'no', values: ['21', '23'] }],
    },
  ],
};

function render(value = report) {
  TestBed.configureTestingModule({ providers: [provideRouter([])] });
  const fixture = TestBed.createComponent(NeighbourExplorer);
  fixture.componentRef.setInput('report', value);
  fixture.detectChanges();
  return fixture;
}

const cells = (row: Element) =>
  [...row.querySelectorAll('td, th')].map((c) => c.textContent?.trim());

describe('NeighbourExplorer', () => {
  it('starts on the biggest miss, with the row above the neighbours that decided it', () => {
    const element = render().nativeElement as HTMLElement;
    const rows = [...element.querySelectorAll('tbody tr')];
    expect(rows).toHaveLength(3);
    expect(cells(rows[0])).toEqual(['This row', '–', '–', 'no', '30', '50']);
    expect(cells(rows[1])).toEqual(['Neighbour 1', '0.5', '75%', 'yes', '31', '49']);
    const headers = [...element.querySelectorAll('th')].map((h) => h.textContent?.trim());
    expect(headers).toEqual(['Row', 'Distance', 'Say', 'Target', 'bmi', 'age']);
    expect(element.querySelector('option')?.textContent).toContain('Big miss: predicted yes');
  });

  it('switches to another test row', async () => {
    const fixture = render();
    const element = fixture.nativeElement as HTMLElement;
    const select = element.querySelector('select') as HTMLSelectElement;
    select.value = '1';
    select.dispatchEvent(new Event('change'));
    await fixture.whenStable();
    const rows = [...element.querySelectorAll('tbody tr')];
    expect(rows).toHaveLength(2);
    expect(cells(rows[1])).toEqual(['Neighbour 1', '0.25', '100%', 'no', '21', '23']);
  });

  it('says how near the neighbours are and how many columns are hidden', () => {
    const element = render().nativeElement as HTMLElement;
    expect(element.textContent).toContain('0.25'); // neighbour ÷ typical
    expect(element.textContent).toContain('3 more also count');
    expect(element.querySelector('a')?.getAttribute('href')).toBe(
      '/concepts/curse-of-dimensionality',
    );
  });

  it('mentions no hidden columns when all fit', () => {
    const element = render({ ...report, total_columns: 2 }).nativeElement as HTMLElement;
    expect(element.textContent).not.toContain('more also count');
  });
});
