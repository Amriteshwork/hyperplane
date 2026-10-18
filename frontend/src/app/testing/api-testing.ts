import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { ApplicationRef, EnvironmentProviders, Provider } from '@angular/core';
import { TestBed } from '@angular/core/testing';

import { provideApi } from '../core/config/api-base-url';

/** The generated client over a mocked backend. */
export function provideApiTesting(): (Provider | EnvironmentProviders)[] {
  return [provideHttpClient(), provideHttpClientTesting(), provideApi()];
}

/**
 * Answers the next request to `url` once it is made (resources fire asynchronously), then
 * lets rendering settle. Not `whenStable()`: that also waits for requests not yet answered.
 */
export async function respond(
  url: string,
  body: object | null,
  options: { status?: number; statusText?: string } = {},
): Promise<void> {
  const http = TestBed.inject(HttpTestingController);
  for (let i = 0; i < 50; i++) {
    const [request] = http.match(url);
    if (request) {
      request.flush(body, options);
      await settle();
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  throw new Error(`no request to ${url}`);
}

/** Runs pending effects and change detection without waiting for open HTTP requests. */
export async function settle(): Promise<void> {
  for (let i = 0; i < 3; i++) {
    await new Promise((resolve) => setTimeout(resolve, 0));
    TestBed.inject(ApplicationRef).tick();
  }
}
