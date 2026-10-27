import { type Page, expect, test } from '@playwright/test';

/**
 * Trains both MLP packs through the real Lab (sample → profile → train → results), so it needs
 * the live stack: `make up`, then `E2E_LIVE=1 E2E_BASE_URL=http://localhost:8080 npm run e2e`.
 */
test.skip(process.env['E2E_LIVE'] !== '1', 'needs the live backend and worker (E2E_LIVE=1)');
test.setTimeout(120_000);

const lossCurve = /^The average loss on the training rows after each pass/;

for (const name of ['Diabetes', 'California Housing']) {
  test(`trains MLP Regressor on ${name}`, async ({ page }) => {
    await page.goto('/lab?algorithm=mlp_regressor');

    await page.getByRole('button', { name: new RegExp(`^${name}`) }).click();
    await page.getByRole('button', { name: 'Next: settings' }).click({ timeout: 60_000 });

    await expect(page.getByLabel('Algorithm')).toHaveValue('mlp_regressor');
    await page.getByLabel('Activation').selectOption('tanh');
    await page.getByRole('button', { name: 'Train model' }).click();

    await expect(page.getByTestId('run-status')).toHaveText('succeeded', { timeout: 90_000 });
    await expect(page.getByText('mlp_regressor ·')).toBeVisible();
    await expect(page.getByRole('heading', { name: /Test-set metrics/ })).toBeVisible();
    await expect(page.getByLabel(lossCurve)).toBeVisible();
    await expect(page.getByRole('table', { name: 'Learned coefficients' })).toHaveCount(0);
    await expectNetworkViews(page);
  });
}

// lbfgs is full-batch: no epochs, so no loss curve.
const CLASSIFIER_RUNS = [
  { name: 'Breast Cancer', solver: 'adam', curve: true },
  { name: 'Iris', solver: 'lbfgs', curve: false },
];

for (const { name, solver, curve } of CLASSIFIER_RUNS) {
  test(`trains MLP Classifier (${solver}) on ${name}`, async ({ page }) => {
    await page.goto('/lab?algorithm=mlp_classifier');

    await page.getByRole('button', { name: new RegExp(`^${name}`) }).click();
    await page.getByRole('button', { name: 'Next: settings' }).click({ timeout: 60_000 });

    await expect(page.getByLabel('Algorithm')).toHaveValue('mlp_classifier');
    await page.getByRole('button', { name: /^Advanced/ }).click();
    await page.getByLabel('Solver').selectOption(solver);
    await page.getByRole('button', { name: 'Train model' }).click();

    await expect(page.getByTestId('run-status')).toHaveText('succeeded', { timeout: 90_000 });
    await expect(page.getByText('mlp_classifier ·')).toBeVisible();
    await expect(page.getByText('Accuracy', { exact: true })).toBeVisible();
    await expect(page.getByLabel(/^Test rows by actual class/)).toBeVisible(); // confusion matrix
    await expect(page.getByLabel(/^Each point is a threshold/)).toBeVisible(); // ROC curve
    const loss = page.getByLabel(lossCurve);
    await (curve ? expect(loss).toBeVisible() : expect(loss).toHaveCount(0));
    await expectNetworkViews(page);
  });
}

/** The diagram of the trained network and its first-layer weights. */
async function expectNetworkViews(page: Page): Promise<void> {
  await expect(page.getByRole('heading', { name: 'The trained network' })).toBeVisible();
  await expect(page.getByRole('img', { name: /^Diagram of the trained network/ })).toBeVisible();
  await expect(page.getByLabel(/^Bar chart of mean absolute first-layer weight/)).toBeVisible();
}
