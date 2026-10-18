import { HttpErrorResponse } from '@angular/common/http';
import { Injectable, computed, inject, resource } from '@angular/core';

import { Api } from '../../core/api/api';
import { getHealth } from '../../core/api/fn/health/get-health';
import { HealthStatus } from '../../core/api/models/health-status';

export interface ServiceRow {
  id: string;
  label: string;
  up: boolean;
  /** Probe latency, or null when unknown. */
  latencyMs: number | null;
  error: string | null;
}

const LABELS: Record<string, string> = {
  api: 'API',
  db: 'PostgreSQL',
  redis: 'Redis',
  storage: 'Object storage',
};

/** Loads /api/v1/health and turns it into display rows. Provided by the home page. */
@Injectable()
export class HealthStore {
  private readonly api = inject(Api);

  private readonly health = resource({ loader: () => this.fetch() });

  readonly loading = this.health.isLoading;

  /** API row first (reachable if we got any answer), then each dependency. */
  readonly rows = computed<ServiceRow[]>(() => {
    if (this.health.isLoading() && !this.health.hasValue()) return [];
    if (this.health.error() || !this.health.hasValue()) {
      return [
        { id: 'api', label: LABELS['api'], up: false, latencyMs: null, error: 'Unreachable' },
      ];
    }
    const services = Object.entries(this.health.value().services).map(([id, s]) => ({
      id,
      label: LABELS[id] ?? id,
      up: s.status === 'up',
      latencyMs: s.latency_ms,
      error: s.error ?? null,
    }));
    return [
      { id: 'api', label: LABELS['api'], up: true, latencyMs: null, error: null },
      ...services,
    ];
  });

  readonly allUp = computed(() => this.rows().length > 0 && this.rows().every((row) => row.up));

  reload(): void {
    this.health.reload();
  }

  /** A 503 still carries a HealthStatus body describing what is down. */
  private async fetch(): Promise<HealthStatus> {
    try {
      return await this.api.invoke(getHealth);
    } catch (error) {
      if (
        error instanceof HttpErrorResponse &&
        error.status === 503 &&
        isHealthStatus(error.error)
      ) {
        return error.error;
      }
      throw error;
    }
  }
}

function isHealthStatus(value: unknown): value is HealthStatus {
  return typeof value === 'object' && value !== null && 'services' in value && 'status' in value;
}
