import { InjectionToken, Provider, inject } from '@angular/core';

import { ApiConfiguration } from '../api/api-configuration';

/** Origin the generated API client calls. Empty = same origin (dev proxy / nginx `/api`). */
export const API_BASE_URL = new InjectionToken<string>('API_BASE_URL', {
  providedIn: 'root',
  factory: () => '',
});

/** Points the generated client at `API_BASE_URL`. */
export function provideApi(): Provider {
  return {
    provide: ApiConfiguration,
    useFactory: () => {
      const config = new ApiConfiguration();
      config.rootUrl = inject(API_BASE_URL);
      return config;
    },
  };
}
