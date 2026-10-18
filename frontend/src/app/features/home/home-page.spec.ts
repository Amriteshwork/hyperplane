import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';

import { HealthStatus } from '../../core/api/models/health-status';
import { HomePage } from './home-page';

const HEALTH_URL = '/api/v1/health';

describe('HomePage system status', () => {
  let fixture: ComponentFixture<HomePage>;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [HomePage],
      providers: [provideRouter([]), provideHttpClient(), provideHttpClientTesting()],
    });
    http = TestBed.inject(HttpTestingController);
    fixture = TestBed.createComponent(HomePage);
    // Start the health request; whenStable() would wait for it, so flush it in each test.
    TestBed.tick();
  });

  afterEach(() => http.verify());

  const text = (testId: string) =>
    (fixture.nativeElement as HTMLElement)
      .querySelector(`[data-testid="${testId}"]`)
      ?.textContent?.replace(/\s+/g, ' ')
      .trim();

  it('shows every service up when the API is healthy', async () => {
    const body: HealthStatus = {
      status: 'ok',
      services: {
        db: { status: 'up', latency_ms: 3 },
        redis: { status: 'up', latency_ms: 1 },
        storage: { status: 'up', latency_ms: 5 },
      },
    };
    http.expectOne(HEALTH_URL).flush(body);
    await fixture.whenStable();

    expect(text('overall-status')).toBe('All systems up');
    expect(text('service-api')).toContain('Up');
    expect(text('service-db')).toContain('PostgreSQL');
  });

  it('reads the 503 body to show which dependency is down', async () => {
    const body: HealthStatus = {
      status: 'degraded',
      services: {
        db: { status: 'down', latency_ms: 2000, error: 'Timeout' },
        redis: { status: 'up', latency_ms: 1 },
        storage: { status: 'up', latency_ms: 5 },
      },
    };
    http.expectOne(HEALTH_URL).flush(body, { status: 503, statusText: 'Service Unavailable' });
    await fixture.whenStable();

    expect(text('overall-status')).toBe('Degraded');
    expect(text('service-api')).toContain('Up');
    expect(text('service-db')).toContain('Down');
  });

  it('marks the API down when it cannot be reached', async () => {
    http.expectOne(HEALTH_URL).error(new ProgressEvent('error'), { status: 0 });
    await fixture.whenStable();

    expect(text('overall-status')).toBe('Degraded');
    expect(text('service-api')).toContain('Down');
  });
});
