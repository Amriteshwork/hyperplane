import { expect, test } from '@playwright/test';

/**
 * Trains both SGD packs through the real Lab (sample → profile → train → results), so it needs
 * the live stack: `make up`, then `E2E_LIVE=1 E2E_BASE_URL=http://localhost:8080 npm run e2e`.
 */
test.skip(process.env['E2E_LIVE'] !== '1', 'needs the live backend and worker (E2E_LIVE=1)');
test.setTimeout(120_000);

// California has rows with extreme features: the case the regressor's small default step is for.
for (const name of ['Diabetes', 'California Housing']) {
  test(`trains SGD Regressor on ${name}`, async ({ page }) => {
    await page.goto('/lab?algorithm=sgd_regressor');

    await page.getByRole('button', { name: new RegExp(`^${name}`) }).click();
    await page.getByRole('button', { name: 'Next: settings' }).click({ timeout: 60_000 });

    await expect(page.getByLabel('Algorithm')).toHaveValue('sgd_regressor');
    await page.getByLabel('Loss').selectOption('huber');
    await page.getByRole('button', { name: 'Train model' }).click();

    await expect(page.getByTestId('run-status')).toHaveText('succeeded', { timeout: 90_000 });
    await expect(page.getByText('sgd_regressor ·')).toBeVisible();
    await expect(page.getByRole('heading', { name: /Test-set metrics/ })).toBeVisible();
    await expect(page.getByRole('table', { name: 'Learned coefficients' })).toBeVisible();
  });
}

// log_loss gives probabilities (and so a ROC curve); hinge gives labels only.
const CLASSIFIER_RUNS = [
  { name: 'Breast Cancer', loss: 'log_loss', roc: true },
  { name: 'Iris', loss: 'hinge', roc: false },
];

for (const { name, loss, roc } of CLASSIFIER_RUNS) {
  test(`trains SGD Classifier (${loss}) on ${name}`, async ({ page }) => {
    await page.goto('/lab?algorithm=sgd_classifier');

    await page.getByRole('button', { name: new RegExp(`^${name}`) }).click();
    await page.getByRole('button', { name: 'Next: settings' }).click({ timeout: 60_000 });

    await expect(page.getByLabel('Algorithm')).toHaveValue('sgd_classifier');
    await page.getByLabel('Loss').selectOption(loss);
    await page.getByRole('button', { name: 'Train model' }).click();

    await expect(page.getByTestId('run-status')).toHaveText('succeeded', { timeout: 90_000 });
    await expect(page.getByText('sgd_classifier ·')).toBeVisible();
    await expect(page.getByText('Accuracy', { exact: true })).toBeVisible();
    await expect(page.getByLabel(/^Test rows by actual class/)).toBeVisible(); // confusion matrix
    const rocCurve = page.getByLabel(/^Each point is a threshold/);
    await (roc ? expect(rocCurve).toBeVisible() : expect(rocCurve).toHaveCount(0));
    const coefficients = page.getByRole('table', { name: 'Learned coefficients' });
    await expect(coefficients.getByRole('columnheader', { name: 'Class' })).toBeVisible();
  });
}
