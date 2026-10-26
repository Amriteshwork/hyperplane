import { ChangeDetectionStrategy, Component, input } from '@angular/core';
import { RouterLink } from '@angular/router';

import { TrackSummaryOut } from '../../core/api/models/track-summary-out';

/** A guided track as a card; used in Learn's family sections and on the concept index. */
@Component({
  selector: 'app-track-card',
  imports: [RouterLink],
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: { class: 'block h-full' },
  template: `
    <a
      class="flex h-full flex-col gap-2 rounded-lg border-2 border-primary/40 bg-card p-4 transition-colors hover:border-primary hover:bg-muted/40 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring motion-reduce:transition-none"
      [routerLink]="['/tracks', track().id]"
    >
      <span class="text-xs font-medium tracking-wide text-muted-foreground uppercase">
        Guided track
      </span>
      <span class="font-semibold">{{ track().title }}</span>
      <span class="text-sm text-muted-foreground">{{ track().summary }}</span>
      <span class="mt-auto pt-1 text-xs text-muted-foreground">
        {{ track().concept_count }} concepts · ends at {{ track().goal.title }}
      </span>
    </a>
  `,
})
export class TrackCard {
  readonly track = input.required<TrackSummaryOut>();
}
