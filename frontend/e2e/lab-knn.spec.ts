import { type Page, expect, test } from '@playwright/test';

/**
 * Trains both KNN packs through the real Lab (sample → profile → train → results), so it needs
 * the live stack: `make up`, then `E2E_LIVE=1 E2E_BASE_URL=http://localhost:8080 npm run e2e`.
 */
test.skip(process.env['E2E_LIVE'] !== '1', 'needs the live backend and worker (E2E_LIVE=1)');
test.setTimeout(120_000);

const tuningCurve = /^Each point refits on parts of the training rows/;

for (const name of ['Diabetes', 'California Housing']) {
  test(`trains KNN Regressor on ${name}`, async ({ page }) => {
    await page.goto('/lab?algorithm=knn_regressor');

    await page.getByRole('button', { name: new RegExp(`^${name}`) }).click();
    await page.getByRole('button', { name: 'Next: settings' }).click({ timeout: 60_000 });

    await expect(page.getByLabel('Algorithm')).toHaveValue('knn_regressor');
    await page.getByLabel('Neighbour weights').selectOption('distance');
    await page.getByRole('button', { name: 'Train model' }).click();

    await expect(page.getByTestId('run-status')).toHaveText('succeeded', { timeout: 90_000 });
    await expect(page.getByText('knn_regressor ·')).toBeVisible();
    await expect(page.getByRole('heading', { name: /Test-set metrics/ })).toBeVisible();
    await expect(page.getByLabel(tuningCurve)).toBeVisible();
    await expect(page.getByRole('table', { name: 'Learned coefficients' })).toHaveCount(0);
    await expectNeighbourViews(page);
  });
}

for (const name of ['Breast Cancer', 'Iris']) {
  test(`trains KNN Classifier on ${name}`, async ({ page }) => {
    await page.goto('/lab?algorithm=knn_classifier');

    await page.getByRole('button', { name: new RegExp(`^${name}`) }).click();
    await page.getByRole('button', { name: 'Next: settings' }).click({ timeout: 60_000 });

    await expect(page.getByLabel('Algorithm')).toHaveValue('knn_classifier');
    await page.getByRole('button', { name: 'Train model' }).click();

    await expect(page.getByTestId('run-status')).toHaveText('succeeded', { timeout: 90_000 });
    await expect(page.getByText('knn_classifier ·')).toBeVisible();
    await expect(page.getByText('Accuracy', { exact: true })).toBeVisible();
    await expect(page.getByLabel(/^Test rows by actual class/)).toBeVisible(); // confusion matrix
    await expect(page.getByLabel(/^Each point is a threshold/)).toBeVisible(); // ROC
    await expect(page.getByLabel(tuningCurve)).toBeVisible();
    await expectNeighbourViews(page);
  });
}

// k = 1 is never the best on California's noisy target, so the suggestion always offers a change.
test('takes up the suggested k and trains with it', async ({ page }) => {
  await page.goto('/lab?algorithm=knn_regressor');
  await page.getByRole('button', { name: /^California Housing/ }).click();
  await page.getByRole('button', { name: 'Next: settings' }).click({ timeout: 60_000 });
  await page.getByLabel('Neighbours (k)').fill('1');
  await page.getByLabel('Neighbour weights').selectOption('distance');
  await page.getByRole('button', { name: 'Train model' }).click();
  await expect(page.getByTestId('run-status')).toHaveText('succeeded', { timeout: 90_000 });

  const suggestion = page.getByRole('button', { name: /^Set up a run with \d+/ });
  const best = (await suggestion.textContent())?.match(/\d+/)?.[0] ?? '';
  await suggestion.click();

  await expect(page.getByLabel('Neighbours (k)')).toHaveValue(best); // the suggestion
  await expect(page.getByLabel('Neighbour weights')).toHaveValue('1'); // distance, kept from the run (options are keyed by index)
  await page.getByRole('button', { name: 'Train model' }).click();
  await expect(page.getByTestId('run-status')).toHaveText('succeeded', { timeout: 90_000 });
  await expect(page.getByText(/already has the best/)).toBeVisible();
});

async function expectNeighbourViews(page: Page): Promise<void> {
  const table = page.getByRole('table', { name: /stored rows behind its prediction/ });
  await expect(table.getByRole('cell', { name: 'This row' })).toBeVisible();
  await expect(table.getByRole('cell', { name: 'Neighbour 5' })).toBeVisible();

  const example = page.getByRole('combobox', { name: 'Test row' });
  await expect(example.locator('option').first()).toContainText('Big miss');
  await example.selectOption({ index: 3 });
  await expect(table.getByRole('cell', { name: 'This row' })).toBeVisible();
  await expect(page.getByText('Neighbour ÷ typical')).toBeVisible();
}
