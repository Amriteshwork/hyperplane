import { expect, test } from '@playwright/test';

/**
 * Trains Polynomial Regression through the real Lab (upload → profile → train → results), so it
 * needs the live stack: `make up`, then `E2E_LIVE=1 E2E_BASE_URL=http://localhost:8080 npm run e2e`.
 */
test.skip(process.env['E2E_LIVE'] !== '1', 'needs the live backend and worker (E2E_LIVE=1)');
test.setTimeout(120_000);

const SAMPLES = [
  { name: 'Diabetes', degree: 2 },
  { name: 'California Housing', degree: 3 },
];

for (const { name, degree } of SAMPLES) {
  test(`trains Polynomial Regression (degree ${degree}) on ${name}`, async ({ page }) => {
    await page.goto('/lab?algorithm=polynomial_regression');

    await page.getByRole('button', { name: new RegExp(`^${name}`) }).click();
    await page.getByRole('button', { name: 'Next: settings' }).click({ timeout: 60_000 });

    await expect(page.getByLabel('Algorithm')).toHaveValue('polynomial_regression');
    await page.getByLabel('Degree').fill(String(degree));
    await page.getByRole('button', { name: 'Train model' }).click();

    await expect(page.getByTestId('run-status')).toHaveText('succeeded', { timeout: 90_000 });
    await expect(page.getByRole('heading', { name: /Test-set metrics/ })).toBeVisible();
    // Expanded terms reach the coefficient table as squares and products of features.
    const coefficients = page.getByRole('table', { name: 'Learned coefficients' });
    await expect(coefficients).toContainText('^2');
    await expect(coefficients).toContainText(' × ');
    if (degree === 3) await expect(coefficients).toContainText('^3');
  });
}
