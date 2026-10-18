import { ChangeDetectionStrategy, Component, input } from '@angular/core';
import { HlmBadge } from '@spartan-ng/helm/badge';

export type TableCell = string | number | boolean | null;

export interface TableColumn {
  readonly label: string;
  /** Shown as a badge under the header, e.g. the inferred column type. */
  readonly badge?: string;
  readonly align?: 'start' | 'end';
}

/** Positional rows (`rows[i][j]` belongs to `columns[j]`), matching the preview API. */
@Component({
  selector: 'app-data-table',
  imports: [HlmBadge],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div
      class="overflow-auto rounded-lg border focus-visible:outline-2 focus-visible:outline-ring"
      tabindex="0"
      role="region"
      [attr.aria-label]="caption()"
      [style.max-height]="maxHeight()"
    >
      <table class="w-full border-collapse text-sm">
        <caption class="sr-only">
          {{
            caption()
          }}
        </caption>
        <thead class="sticky top-0 z-10 bg-muted/95 backdrop-blur-sm">
          <tr>
            @for (column of columns(); track $index) {
              <th
                scope="col"
                class="border-b px-3 py-2 font-medium whitespace-nowrap"
                [class.text-right]="column.align === 'end'"
                [class.text-left]="column.align !== 'end'"
              >
                <span class="block">{{ column.label }}</span>
                @if (column.badge) {
                  <span hlmBadge variant="outline" class="mt-1 font-normal">{{
                    column.badge
                  }}</span>
                }
              </th>
            }
          </tr>
        </thead>
        <tbody>
          @for (row of rows(); track $index) {
            <tr class="border-b last:border-0 hover:bg-muted/40">
              @for (cell of row; track $index) {
                <td
                  class="px-3 py-1.5 whitespace-nowrap"
                  [class.text-right]="columns()[$index]?.align === 'end'"
                  [class.tabular-nums]="columns()[$index]?.align === 'end'"
                >
                  @if (cell === null) {
                    <span class="text-muted-foreground italic">missing</span>
                  } @else {
                    {{ cell }}
                  }
                </td>
              }
            </tr>
          } @empty {
            <tr>
              <td
                class="px-3 py-6 text-center text-muted-foreground"
                [attr.colspan]="columns().length || 1"
              >
                {{ emptyText() }}
              </td>
            </tr>
          }
        </tbody>
      </table>
    </div>
  `,
})
export class DataTable {
  readonly columns = input.required<readonly TableColumn[]>();
  readonly rows = input.required<readonly (readonly TableCell[])[]>();
  /** Accessible name for the table and its scroll region. */
  readonly caption = input.required<string>();
  readonly maxHeight = input('28rem');
  readonly emptyText = input('No rows.');
}
