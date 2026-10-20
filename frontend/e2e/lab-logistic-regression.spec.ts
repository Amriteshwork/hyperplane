import { expect, test } from '@playwright/test';

/**
 * Trains Logistic Regression through the real Lab (sample → profile → train → results), so it
 * needs the live stack: `make up`, then `E2E_LIVE=1 E2E_BASE_URL=http://localhost:8080 npm run e2e`.
 */
test.skip(process.env['E2E_LIVE'] !== '1', 'needs the live backend and worker (E2E_LIVE=1)');
test.setTimeout(120_000);

// One binary and one multiclass sample: the training script and metrics differ between them.
const SAMPLES = [
  { name: 'Breast Cancer', precision: 'Precision (malignant)', classWeight: 'balanced' },
  { name: 'Iris', precision: 'Precision (macro)', classWeight: 'None' },
];

for (const { name, precision, classWeight } of SAMPLES) {
  test(`trains Logistic Regression on ${name}`, async ({ page }) => {
    await page.goto('/lab?algorithm=logistic_regression');

    await page.getByRole('button', { name: new RegExp(`^${name}`) }).click();
    await page.getByRole('button', { name: 'Next: settings' }).click({ timeout: 60_000 });

    await expect(page.getByLabel('Algorithm')).toHaveValue('logistic_regression');
    await page.getByLabel('Inverse regularisation strength (C)').fill('0.5');
    await page.getByLabel('Class weights').selectOption({ label: classWeight });
    await page.getByRole('button', { name: 'Train model' }).click();

    await expect(page.getByTestId('run-status')).toHaveText('succeeded', { timeout: 90_000 });
    await expect(page.getByText('logistic_regression ·')).toBeVisible();
    await expect(page.getByText('Accuracy', { exact: true })).toBeVisible();
    await expect(page.getByText(precision, { exact: true })).toBeVisible();
    await expect(page.getByLabel(/^Test rows by actual class/)).toBeVisible(); // confusion matrix
    await expect(page.getByLabel(/^Each point is a threshold/)).toBeVisible(); // ROC
    const coefficients = page.getByRole('table', { name: 'Learned coefficients' });
    await expect(coefficients.getByRole('columnheader', { name: 'Class' })).toBeVisible();
  });
}
