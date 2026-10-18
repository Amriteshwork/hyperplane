import { defineConfig, devices } from '@playwright/test';

/**
 * E2E_BASE_URL targets an already running app (e.g. the Docker stack on :8080);
 * otherwise Playwright starts `ng serve` on :4200.
 */
const baseURL = process.env['E2E_BASE_URL'] ?? 'http://localhost:4200';
const isCI = !!process.env['CI'];

export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: isCI,
  retries: isCI ? 2 : 0,
  reporter: isCI ? [['github'], ['html', { open: 'never' }]] : 'list',
  use: { baseURL, trace: 'on-first-retry' },
  projects: [
    { name: 'desktop', use: { ...devices['Desktop Chrome'] } },
    { name: 'mobile', use: { ...devices['Pixel 7'] } },
  ],
  webServer: process.env['E2E_BASE_URL']
    ? undefined
    : {
        // No Vite prebundling: on a cold cache it reloads the page mid-test.
        command: 'npx ng serve --no-prebundle',
        url: baseURL,
        reuseExistingServer: !isCI,
        timeout: 180_000,
      },
});
