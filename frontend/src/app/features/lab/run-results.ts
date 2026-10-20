import { ChangeDetectionStrategy, Component, computed, input } from '@angular/core';

import { TrainingReport } from '../../core/api/models/training-report';
import { Chart } from '../../shared/ui/chart/chart';
import { DataTable, TableColumn } from '../../shared/ui/data-table/data-table';
import { StatCard } from '../../shared/ui/stat-card/stat-card';
import { plotOptions } from './plot-options';

const COEFFICIENT_COLUMNS: readonly TableColumn[] = [
  { label: 'Feature' },
  { label: 'Coefficient', align: 'end' },
];
const CLASS_COEFFICIENT_COLUMNS: readonly TableColumn[] = [
  { label: 'Class' },
  ...COEFFICIENT_COLUMNS,
];

/** Metrics, evaluation plots and coefficients of a finished run. */
@Component({
  selector: 'app-run-results',
  imports: [Chart, DataTable, StatCard],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <section aria-labelledby="metrics-heading">
      <h2 id="metrics-heading" class="text-lg font-semibold">
        Test-set metrics
        <span class="text-sm font-normal text-muted-foreground">
          ({{ report().n_test }} held-out rows; trained on {{ report().n_train }})
        </span>
      </h2>
      <ul class="mt-3 grid gap-3 sm:grid-cols-3">
        @for (metric of report().evaluation.metrics; track metric.id) {
          <li>
            <app-stat-card
              class="h-full"
              [label]="metric.label"
              [value]="format(metric.value)"
              [hint]="metric.description"
            />
          </li>
        }
      </ul>
    </section>

    <section class="mt-10" aria-labelledby="plots-heading">
      <h2 id="plots-heading" class="text-lg font-semibold">How the predictions look</h2>
      <div class="mt-3 grid gap-6 lg:grid-cols-2">
        @for (plot of plots(); track plot.id) {
          <app-chart
            [caption]="plot.title"
            [label]="plot.description"
            [options]="plot.options"
            [dark]="dark()"
          >
            <p class="text-xs text-muted-foreground">{{ plot.description }}</p>
          </app-chart>
        }
      </div>
    </section>

    @if (coefficientRows().length) {
      <section class="mt-10" aria-labelledby="coefficients-heading">
        <h2 id="coefficients-heading" class="text-lg font-semibold">Coefficients</h2>
        <p class="mt-1 max-w-prose text-sm text-muted-foreground">
          The weight learned for each feature after scaling (or for each power or product of
          features, written a^2 or a × b), largest effect first.
          @if (perClass()) {
            Each class has its own weights: positive values raise the log-odds of that class.
          } @else {
            Positive values push the prediction up.
          }
        </p>
        <app-data-table
          class="mt-3 block"
          caption="Learned coefficients"
          [columns]="perClass() ? classCoefficientColumns : coefficientColumns"
          [rows]="coefficientRows()"
        />
      </section>
    }
  `,
})
export class RunResults {
  readonly report = input.required<TrainingReport>();
  readonly dark = input(false);

  protected readonly coefficientColumns = COEFFICIENT_COLUMNS;
  protected readonly classCoefficientColumns = CLASS_COEFFICIENT_COLUMNS;
  protected readonly plots = computed(() =>
    this.report().evaluation.plots.map((plot) => ({ ...plot, options: plotOptions(plot) })),
  );
  /** Classifiers report one weight per class and feature. */
  protected readonly perClass = computed(() =>
    (this.report().coefficients ?? []).some((c) => c.target_class != null),
  );
  protected readonly coefficientRows = computed(() =>
    (this.report().coefficients ?? []).map((c) => {
      const row = [featureLabel(c.feature), this.format(c.value)];
      return this.perClass() ? [c.target_class ?? '', ...row] : row;
    }),
  );

  protected format(value: number): string {
    return Number(value.toPrecision(4)).toLocaleString('en', { maximumFractionDigits: 4 });
  }
}

/**
 * `numeric__age` → `age`, and a product term `numeric__age numeric__bmi` → `age × bmi`: drop each
 * preprocessing prefix. Only a space before another `step__` prefix separates factors, since column
 * names may contain spaces.
 */
export function featureLabel(name: string): string {
  return name
    .split(/ (?=[a-z]+__)/)
    .map((factor) => {
      const split = factor.indexOf('__');
      return split === -1 ? factor : factor.slice(split + 2);
    })
    .join(' × ');
}
