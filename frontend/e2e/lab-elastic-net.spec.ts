import { expect, test } from '@playwright/test';

/**
 * Trains Elastic Net through the real Lab (sample → profile → train → results), so it needs the
 * live stack: `make up`, then `E2E_LIVE=1 E2E_BASE_URL=http://localhost:8080 npm run e2e`.
 */
test.skip(process.env['E2E_LIVE'] !== '1', 'needs the live backend and worker (E2E_LIVE=1)');
test.setTimeout(120_000);

// α is in the target's units: Diabetes' target varies by ~77, California's by ~1.
const SAMPLES = [
  { name: 'Diabetes', alpha: 1, l1Ratio: 0.9 },
  { name: 'California Housing', alpha: 0.1, l1Ratio: 0.5 },
];

for (const { name, alpha, l1Ratio } of SAMPLES) {
  test(`trains Elastic Net (α = ${alpha}, ρ = ${l1Ratio}) on ${name}`, async ({ page }) => {
    await page.goto('/lab?algorithm=elastic_net');

    await page.getByRole('button', { name: new RegExp(`^${name}`) }).click();
    await page.getByRole('button', { name: 'Next: settings' }).click({ timeout: 60_000 });

    await expect(page.getByLabel('Algorithm')).toHaveValue('elastic_net');
    await page.getByLabel('Regularisation strength (α)').fill(String(alpha));
    await page.getByLabel('L1 ratio (ρ)').fill(String(l1Ratio));
    await page.getByRole('button', { name: 'Train model' }).click();

    await expect(page.getByTestId('run-status')).toHaveText('succeeded', { timeout: 90_000 });
    await expect(page.getByText(`elastic_net ·`)).toBeVisible();
    await expect(page.getByRole('heading', { name: /Test-set metrics/ })).toBeVisible();
    await expect(page.getByRole('table', { name: 'Learned coefficients' })).toBeVisible();
  });
}
