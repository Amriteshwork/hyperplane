import { TestBed } from '@angular/core/testing';

import { ServiceRow } from './health.store';
import { ServiceStatusList } from './service-status-list';

describe('ServiceStatusList', () => {
  it('shows each service with an up/down badge and latency', async () => {
    const rows: ServiceRow[] = [
      { id: 'db', label: 'PostgreSQL', up: true, latencyMs: 4.2, error: null },
      { id: 'redis', label: 'Redis', up: false, latencyMs: 2000, error: 'Timeout' },
    ];
    const fixture = TestBed.createComponent(ServiceStatusList);
    fixture.componentRef.setInput('rows', rows);
    await fixture.whenStable();
    const el: HTMLElement = fixture.nativeElement;

    const db = el.querySelector('[data-testid="service-db"]');
    expect(db?.textContent).toContain('PostgreSQL');
    expect(db?.textContent).toContain('4.2 ms');
    expect(db?.textContent).toContain('Up');

    const redis = el.querySelector('[data-testid="service-redis"] [data-slot="badge"]');
    expect(redis?.textContent?.trim()).toBe('Down');
    expect(redis?.getAttribute('title')).toBe('Timeout');
  });
});
