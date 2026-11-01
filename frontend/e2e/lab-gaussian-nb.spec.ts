import { type Page, expect, test } from '@playwright/test';

/**
 * Trains Gaussian Naive Bayes through the real Lab (sample → profile → train → results), so it needs
 * the live stack: `make up`, then `E2E_LIVE=1 E2E_BASE_URL=http://localhost:8080 npm run e2e`.
 */
test.skip(process.env['E2E_LIVE'] !== '1', 'needs the live backend and worker (E2E_LIVE=1)');
test.setTimeout(120_000);

const tuningCurve = /^Each point refits on parts of the training rows/;

for (const name of ['Iris', 'Breast Cancer', 'Wine']) {
  test(`trains Gaussian Naive Bayes on ${name}`, async ({ page }) => {
    await trainOn(page, name);
    await expect(page.getByText('gaussian_nb ·')).toBeVisible();
    await expect(page.getByText('Accuracy', { exact: true }).first()).toBeVisible();
    await expect(page.getByLabel(/^Test rows by actual class/)).toBeVisible(); // confusion matrix
    await expect(page.getByLabel(tuningCurve)).toBeVisible();
    await expect(page.getByRole('table', { name: 'Learned coefficients' })).toHaveCount(0);

    await expect(page.getByRole('heading', { name: 'What the model learned' })).toBeVisible();
    await expect(page.getByLabel(/^Bell curves of /)).toBeVisible();
    const feature = page.getByRole('combobox', { name: 'Feature' });
    await feature.selectOption({ index: 1 });
    await expect(page.getByLabel(/^Bell curves of /)).toBeVisible();

    const sum = page.getByRole('table', { name: /How the prediction adds up/ });
    await expect(sum.getByRole('cell', { name: 'Prior' })).toBeVisible();
    await expect(sum.getByRole('cell', { name: 'Total' })).toBeVisible();
    const example = page.getByRole('combobox', { name: 'Test row' });
    await expect(example.locator('option').first()).toContainText('Big miss');
    await example.selectOption({ index: 3 });
    await expect(sum.getByRole('cell', { name: 'Total' })).toBeVisible();
    await expect(page.getByText('Average confidence')).toBeVisible();
  });
}

// One-hot columns that are constant within a class: the case the smoothing curve exists for.
test('Penguins recovers when the suggested smoothing is taken up', async ({ page }) => {
  await trainOn(page, 'Palmer Penguins', '0.000000001');
  const suggestion = page.getByRole('button', { name: /^Set up a run with / });
  const best = (await suggestion.textContent())?.replace('Set up a run with', '').trim() ?? '';
  expect(Number(best)).toBeGreaterThan(1e-9);
  await suggestion.click();

  await expect(page.getByLabel('Variance smoothing')).toHaveValue(best);
  await page.getByRole('button', { name: 'Train model' }).click();
  await expect(page.getByTestId('run-status')).toHaveText('succeeded', { timeout: 90_000 });
  await expect(page.getByText(/already has the best/)).toBeVisible();
});

async function trainOn(page: Page, name: string, smoothing?: string): Promise<void> {
  await page.goto('/lab?algorithm=gaussian_nb');
  await page.getByRole('button', { name: new RegExp(`^${name}`) }).click();
  await page.getByRole('button', { name: 'Next: settings' }).click({ timeout: 60_000 });
  await expect(page.getByLabel('Algorithm')).toHaveValue('gaussian_nb');
  if (smoothing) await page.getByLabel('Variance smoothing').fill(smoothing);
  await page.getByRole('button', { name: 'Train model' }).click();
  await expect(page.getByTestId('run-status')).toHaveText('succeeded', { timeout: 90_000 });
}
