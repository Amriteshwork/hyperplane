import { ChangeDetectionStrategy, Component, computed, input, linkedSignal } from '@angular/core';
import { RouterLink } from '@angular/router';

import { NeighbourReport } from '../../core/api/models/neighbour-report';
import { DataTable, TableColumn } from '../../shared/ui/data-table/data-table';
import { StatCard } from '../../shared/ui/stat-card/stat-card';
import { CONTROL_CLASS } from '../../shared/ui/json-schema-form/widgets/control-class';

/** The model's biggest misses and best predictions, each traced to the rows that decided it. */
@Component({
  selector: 'app-neighbour-explorer',
  imports: [RouterLink, DataTable, StatCard],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <p class="mt-1 max-w-prose text-sm text-muted-foreground">
      Nothing was learned here: the model is {{ report().stored_rows.toLocaleString('en') }} stored
      rows, and each prediction comes from the {{ report().k }} closest. Pick a test row to see
      them.
    </p>

    <div class="mt-3 grid gap-1">
      <label for="neighbour-example" class="text-sm font-medium">Test row</label>
      <select
        id="neighbour-example"
        class="max-w-md"
        [class]="controlClass"
        (change)="selected.set(+$any($event.target).value)"
      >
        @for (example of report().examples; track $index) {
          <option [value]="$index" [selected]="$index === selected()">
            {{ example.kind === 'miss' ? 'Big miss' : 'Good' }}: predicted {{ example.predicted }},
            actual {{ example.actual }}
          </option>
        }
      </select>
    </div>

    <app-data-table
      class="mt-3 block"
      caption="A test row and the stored rows behind its prediction"
      [columns]="columns()"
      [rows]="rows()"
    />
    <p class="mt-2 text-xs text-muted-foreground">
      Distances are measured after scaling, so they aren't in the units shown.
      @if (hiddenColumns(); as hidden) {
        Showing the {{ report().columns.length }} columns where neighbours differ most from their
        query; {{ hidden }} more also count towards each distance.
      }
    </p>

    <h3 class="mt-6 text-sm font-medium">How near is "nearest" on this data?</h3>
    <ul class="mt-2 grid gap-3 sm:grid-cols-3">
      <li>
        <app-stat-card
          class="h-full"
          [label]="'Distance to neighbour ' + report().k"
          [value]="format(report().nearest_distance)"
          hint="On average, to the farthest row that gets a say."
        />
      </li>
      <li>
        <app-stat-card
          class="h-full"
          label="Distance to a typical row"
          [value]="format(report().typical_distance)"
          hint="On average, to any stored row."
        />
      </li>
      <li>
        <app-stat-card
          class="h-full"
          label="Neighbour ÷ typical"
          [value]="ratio()"
          hint="Near 0: neighbours are truly close. Near 1: they are no closer than any other row."
        />
      </li>
    </ul>
    <p class="mt-2 max-w-prose text-xs text-muted-foreground">
      As features are added, this ratio climbs towards 1. That's the
      <a class="underline" routerLink="/concepts/curse-of-dimensionality">curse of dimensionality</a
      >. If it is high, try removing uninformative columns before tuning k.
    </p>
  `,
})
export class NeighbourExplorer {
  readonly report = input.required<NeighbourReport>();

  protected readonly controlClass = CONTROL_CLASS;
  /** Back to the biggest miss whenever a new run's report arrives. */
  protected readonly selected = linkedSignal({ source: this.report, computation: () => 0 });

  private readonly example = computed(() => this.report().examples[this.selected()]);

  protected readonly columns = computed<TableColumn[]>(() => [
    { label: 'Row' },
    { label: 'Distance', align: 'end' },
    { label: 'Say', align: 'end' },
    { label: 'Target' },
    ...this.report().columns.map((label) => ({ label })),
  ]);
  protected readonly rows = computed(() => {
    const { neighbours, actual, values } = this.example();
    return [
      ['This row', '–', '–', actual, ...values],
      ...neighbours.map((n, i) => [
        `Neighbour ${i + 1}`,
        this.format(n.distance),
        `${Math.round(100 * n.share)}%`,
        n.target,
        ...n.values,
      ]),
    ];
  });
  protected readonly hiddenColumns = computed(
    () => this.report().total_columns - this.report().columns.length,
  );
  protected readonly ratio = computed(() => {
    const { nearest_distance: nearest, typical_distance: typical } = this.report();
    return typical ? (nearest / typical).toFixed(2) : '–';
  });

  protected format(value: number): string {
    return Number(value.toPrecision(3)).toLocaleString('en', { maximumFractionDigits: 3 });
  }
}
