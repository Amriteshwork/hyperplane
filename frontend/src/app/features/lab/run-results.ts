import { DecimalPipe } from '@angular/common';
import { ChangeDetectionStrategy, Component, computed, input, output } from '@angular/core';
import { HlmButton } from '@spartan-ng/helm/button';

import { TrainingReport } from '../../core/api/models/training-report';
import { Chart } from '../../shared/ui/chart/chart';
import { DataTable, TableColumn } from '../../shared/ui/data-table/data-table';
import { StatCard } from '../../shared/ui/stat-card/stat-card';
import { featureLabel } from './feature-label';
import { NeighbourExplorer } from './neighbour-explorer';
import { NetworkDiagram } from './network-diagram';
import { barOptions, plotOptions } from './plot-options';

const COEFFICIENT_COLUMNS: readonly TableColumn[] = [
  { label: 'Feature' },
  { label: 'Coefficient', align: 'end' },
];
/** Bars drawn in the importance chart; the rest are listed as a count. */
const MAX_BARS = 15;
const CLASS_COEFFICIENT_COLUMNS: readonly TableColumn[] = [
  { label: 'Class' },
  ...COEFFICIENT_COLUMNS,
];

/** Metrics, evaluation plots, coefficients and whatever else this kind of model can show of a run. */
@Component({
  selector: 'app-run-results',
  imports: [DecimalPipe, HlmButton, Chart, DataTable, NeighbourExplorer, NetworkDiagram, StatCard],
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
      @if (report().tuning; as tuning) {
        <div class="mt-4 flex flex-wrap items-center gap-3 rounded-lg border p-3" role="status">
          @if (tuning.best === tuning.chosen) {
            <p class="text-sm">
              This run's {{ tuning.label }} of {{ tuning.best }} already has the best
              {{ tuning.score_label }} ({{ format(tuning.score) }}) of the values tried.
            </p>
          } @else {
            <p class="text-sm">
              {{ tuning.label }} = {{ tuning.best }} has the best {{ tuning.score_label }} ({{
                format(tuning.score)
              }}) of the values tried.
            </p>
            <button
              hlmBtn
              variant="outline"
              size="sm"
              type="button"
              (click)="tryParameter.emit({ name: tuning.name, value: tuning.best })"
            >
              Set up a run with {{ tuning.best }}
            </button>
          }
        </div>
      }
    </section>

    @if (report().neighbours; as neighbours) {
      <section class="mt-10" aria-labelledby="neighbours-heading">
        <h2 id="neighbours-heading" class="text-lg font-semibold">The rows behind a prediction</h2>
        <app-neighbour-explorer [report]="neighbours" />
      </section>
    }

    @if (report().network; as network) {
      <section class="mt-10" aria-labelledby="network-heading">
        <h2 id="network-heading" class="text-lg font-semibold">The trained network</h2>
        <p class="mt-1 max-w-prose text-sm text-muted-foreground">
          {{ shape() }}: {{ network.parameters | number }} weights and biases, all learned by
          backpropagation.
        </p>
        <app-network-diagram class="mt-3 block" [network]="network" />
      </section>
    }

    @if (importance().length) {
      <section class="mt-10" aria-labelledby="importance-heading">
        <h2 id="importance-heading" class="text-lg font-semibold">
          What the first layer listens to
        </h2>
        <p class="mt-1 max-w-prose text-sm text-muted-foreground">
          The average size of the weights from each input into the first hidden layer. Inputs are
          scaled first, so they compare, but a one-hot column (0 or 1) is on a different scale. This
          shows what the network reads most, not which way a feature moves the prediction.
        </p>
        <app-chart
          class="mt-3 block"
          caption="Mean |weight| into the first hidden layer"
          [label]="importanceLabel()"
          [options]="importanceOptions()"
          [height]="importanceHeight()"
          [dark]="dark()"
        >
          @if (importanceTotal() > importance().length) {
            <p class="text-xs text-muted-foreground">
              The {{ importance().length }} largest of {{ importanceTotal() }} inputs.
            </p>
          }
        </app-chart>
      </section>
    }

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
  /** The learner took up a suggested hyperparameter value. */
  readonly tryParameter = output<{ name: string; value: number }>();

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

  protected readonly shape = computed(() =>
    (this.report().network?.layers ?? [])
      .map((l) => (l.kind === 'hidden' ? `${l.units} ${l.activation}` : `${l.units}`))
      .join(' → '),
  );
  protected readonly importance = computed(() =>
    (this.report().feature_importance ?? [])
      .slice(0, MAX_BARS)
      .map((i) => ({ label: featureLabel(i.feature), value: i.value })),
  );
  protected readonly importanceTotal = computed(
    () => this.report().feature_importance?.length ?? 0,
  );
  protected readonly importanceOptions = computed(() =>
    barOptions(this.importance(), 'Mean |weight|'),
  );
  protected readonly importanceHeight = computed(() => `${80 + 24 * this.importance().length}px`);
  protected readonly importanceLabel = computed(
    () =>
      'Bar chart of mean absolute first-layer weight per input: ' +
      this.importance()
        .map((i) => `${i.label} ${this.format(i.value)}`)
        .join(', '),
  );

  protected format(value: number): string {
    return Number(value.toPrecision(4)).toLocaleString('en', { maximumFractionDigits: 4 });
  }
}
