import { expect, test } from '@playwright/test';

/** E2E_LIVE=1 uses the real backend; otherwise /api/v1/health is stubbed. */
const live = process.env['E2E_LIVE'] === '1';

test.beforeEach(async ({ page }) => {
  if (live) return;
  await page.route('**/api/v1/health', (route) =>
    route.fulfill({
      json: {
        status: 'ok',
        services: {
          db: { status: 'up', latency_ms: 3 },
          redis: { status: 'up', latency_ms: 1 },
          storage: { status: 'up', latency_ms: 4 },
        },
      },
    }),
  );
});

test('home page shows every service up', async ({ page }) => {
  await page.goto('/');

  await expect(page).toHaveTitle('Hyperplane');
  await expect(page.getByRole('heading', { level: 1 })).toContainText('Learn machine learning');
  await expect(page.getByTestId('overall-status')).toHaveText('All systems up');
  for (const id of ['api', 'db', 'redis', 'storage']) {
    await expect(page.getByTestId(`service-${id}`)).toContainText('Up');
  }
});

test('navigates to the Lab', async ({ page, isMobile }) => {
  await page.goto('/');

  if (isMobile) {
    await page.getByTestId('mobile-menu-trigger').click();
    await page
      .getByRole('navigation', { name: 'Mobile' })
      .getByRole('link', { name: 'Lab' })
      .click();
  } else {
    await page
      .getByRole('navigation', { name: 'Primary' })
      .getByRole('link', { name: 'Lab' })
      .click();
  }

  await expect(page).toHaveURL(/\/lab$/);
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('Lab');
  await expect(page).toHaveTitle('Lab · Hyperplane');
});

test('theme toggle switches to dark and persists', async ({ page }) => {
  await page.emulateMedia({ colorScheme: 'light' });
  await page.goto('/');
  await expect(page.locator('html')).not.toHaveClass(/dark/);

  await page.getByTestId('theme-toggle').click();
  await expect(page.locator('html')).toHaveClass(/dark/);

  await page.reload();
  await expect(page.locator('html')).toHaveClass(/dark/);
});
