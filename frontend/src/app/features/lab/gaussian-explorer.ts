import { ChangeDetectionStrategy, Component, computed, input, linkedSignal } from '@angular/core';
import { RouterLink } from '@angular/router';

import { Evidence } from '../../core/api/models/evidence';
import { GaussianReport } from '../../core/api/models/gaussian-report';
import { Chart } from '../../shared/ui/chart/chart';
import { DataTable, TableColumn } from '../../shared/ui/data-table/data-table';
import { CONTROL_CLASS } from '../../shared/ui/json-schema-form/widgets/control-class';
import { StatCard } from '../../shared/ui/stat-card/stat-card';
import { featureLabel } from './feature-label';
import { bellOptions } from './plot-options';

/** Confidence this far above accuracy is worth pointing out. */
const OVERCONFIDENT_BY = 0.02;

/** A per-class Gaussian model opened up: its bell curves, a prediction's sum, and its confidence. */
@Component({
  selector: 'app-gaussian-explorer',
  imports: [RouterLink, Chart, DataTable, StatCard],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <p class="mt-1 max-w-prose text-sm text-muted-foreground">
      The whole model is each class's share of the training rows, plus one bell curve per class and
      feature. Priors: {{ priors() }}.
    </p>

    <div class="mt-3 grid gap-1">
      <label for="gaussian-feature" class="text-sm font-medium">Feature</label>
      <select
        id="gaussian-feature"
        class="max-w-md"
        [class]="controlClass"
        (change)="feature.set(+$any($event.target).value)"
      >
        @for (f of report().features; track f.feature) {
          <option [value]="$index" [selected]="$index === feature()">{{ label(f.feature) }}</option>
        }
      </select>
    </div>
    <app-chart
      class="mt-3 block"
      [caption]="'Bell curves per class for ' + chosenLabel()"
      [label]="bellLabel()"
      [options]="bellOptions()"
      [dark]="dark()"
    >
      <p class="text-xs text-muted-foreground">
        Curves that overlap can't tell the classes apart; curves apart can. {{ listing() }} A
        one-hot column is only ever 0 or 1, so its curve is a rough fit.
      </p>
    </app-chart>

    <h3 class="mt-6 text-sm font-medium">Why this prediction?</h3>
    <p class="mt-1 max-w-prose text-sm text-muted-foreground">
      Each class's score is its log prior plus a log likelihood per feature, and the higher total
      wins. Higher means less surprising.
    </p>
    <div class="mt-3 grid gap-1">
      <label for="gaussian-example" class="text-sm font-medium">Test row</label>
      <select
        id="gaussian-example"
        class="max-w-md"
        [class]="controlClass"
        (change)="selected.set(+$any($event.target).value)"
      >
        @for (e of report().examples; track $index) {
          <option [value]="$index" [selected]="$index === selected()">
            {{ e.kind === 'miss' ? 'Big miss' : 'Good' }}: predicted {{ e.predicted }}, actual
            {{ e.actual }}
          </option>
        }
      </select>
    </div>
    <app-data-table
      class="mt-3 block"
      caption="How the prediction adds up, for the predicted class and its rival"
      [columns]="columns()"
      [rows]="rows()"
    />
    <p class="mt-2 text-xs text-muted-foreground">
      The model gave {{ example().predicted }} {{ percent(example().confidence) }}. Values are what
      the model saw, so a missing value shows what it was filled in with.
    </p>

    <h3 class="mt-6 text-sm font-medium">How far to trust its probabilities?</h3>
    <ul class="mt-2 grid gap-3 sm:grid-cols-3">
      <li>
        <app-stat-card
          class="h-full"
          label="Average confidence"
          [value]="percent(report().mean_confidence)"
          hint="The probability it gave its chosen class, averaged over the test rows."
        />
      </li>
      <li>
        <app-stat-card
          class="h-full"
          label="Accuracy"
          [value]="percent(report().accuracy)"
          hint="How often the chosen class was right. Honest probabilities would match the confidence."
        />
      </li>
      @if (report().correlated; as pair) {
        <li>
          <app-stat-card
            class="h-full"
            label="Most correlated within a class"
            [value]="pair.correlation.toFixed(2)"
            [hint]="
              label(pair.a) + ' and ' + label(pair.b) + '. The model treats them as unrelated.'
            "
          />
        </li>
      }
    </ul>
    <p class="mt-2 max-w-prose text-xs text-muted-foreground">
      @if (overconfident()) {
        It is more sure of itself than it has earned.
      } @else {
        Its confidence is in line with its accuracy on these rows.
      }
      Correlated features count the same evidence twice, which inflates confidence more than it
      harms the labels: see
      <a class="underline" routerLink="/concepts/conditional-independence"
        >conditional independence</a
      >.
    </p>
  `,
})
export class GaussianExplorer {
  readonly report = input.required<GaussianReport>();
  readonly dark = input(false);

  protected readonly controlClass = CONTROL_CLASS;
  protected readonly label = featureLabel;
  /** Back to the first feature and the biggest miss whenever a new run's report arrives. */
  protected readonly feature = linkedSignal({ source: this.report, computation: () => 0 });
  protected readonly selected = linkedSignal({ source: this.report, computation: () => 0 });

  protected readonly priors = computed(() => {
    const { classes, priors } = this.report();
    return classes.map((c, i) => `${c} ${this.percent(priors[i])}`).join(', ');
  });

  protected readonly listing = computed(() => {
    const { features, total_features: total } = this.report();
    const top = total > features.length ? ` (the top ${features.length} of ${total})` : '';
    return `Features are listed from most to least separating${top}.`;
  });

  private readonly chosen = computed(() => this.report().features[this.feature()]);
  protected readonly chosenLabel = computed(() => featureLabel(this.chosen().feature));
  protected readonly bellOptions = computed(() => {
    const { means, stds } = this.chosen();
    return bellOptions(this.report().classes, means, stds, this.chosenLabel());
  });
  protected readonly bellLabel = computed(() => {
    const { means, stds } = this.chosen();
    const curves = this.report().classes.map(
      (c, i) => `${c}: mean ${this.format(means[i])}, spread ${this.format(stds[i])}`,
    );
    return `Bell curves of ${this.chosenLabel()} per class. ${curves.join('; ')}.`;
  });

  protected readonly example = computed(() => this.report().examples[this.selected()]);
  protected readonly columns = computed<TableColumn[]>(() => {
    const { predicted, rival } = this.example();
    return [
      { label: 'Feature' },
      { label: 'Value', align: 'end' },
      { label: `${predicted} (predicted)`, align: 'end' },
      { label: `${rival} (rival)`, align: 'end' },
    ];
  });
  protected readonly rows = computed(() => {
    const { prior, evidence, others } = this.example();
    const entries: Evidence[] = [prior, ...evidence, ...(others ? [others] : [])];
    const sum = (key: 'predicted' | 'rival') => entries.reduce((total, e) => total + e[key], 0);
    return [
      ...entries.map((e) => [
        featureLabel(e.feature),
        e.value,
        this.format(e.predicted),
        this.format(e.rival),
      ]),
      ['Total', '', this.format(sum('predicted')), this.format(sum('rival'))],
    ];
  });
  protected readonly overconfident = computed(
    () => this.report().mean_confidence > this.report().accuracy + OVERCONFIDENT_BY,
  );

  protected percent(value: number): string {
    return `${(100 * value).toFixed(1)}%`;
  }

  protected format(value: number): string {
    return Number(value.toPrecision(4)).toLocaleString('en', { maximumFractionDigits: 3 });
  }
}
