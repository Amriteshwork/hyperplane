import { ChangeDetectionStrategy, Component, input } from '@angular/core';
import { RouterLink } from '@angular/router';
import { HlmBadge } from '@spartan-ng/helm/badge';

import { FamilyOut } from '../../core/api/models/family-out';
import { TASK_LABELS } from '../../shared/learn/labels';
import { TrackCard } from '../../shared/learn/track-card';

/** One family: its guided tracks, then algorithms (available ones link; upcoming are greyed). */
@Component({
  selector: 'app-family-section',
  imports: [RouterLink, HlmBadge, TrackCard],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <section [attr.aria-labelledby]="'family-' + family().id">
      <h2 class="text-xl font-semibold tracking-tight" [id]="'family-' + family().id">
        {{ family().title }}
      </h2>
      <p class="mt-1 max-w-prose text-sm text-muted-foreground">{{ family().summary }}</p>
      <ul class="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        @for (track of family().tracks; track track.id) {
          <li><app-track-card [track]="track" /></li>
        }
        @for (algorithm of family().algorithms; track algorithm.id) {
          <li>
            @if (algorithm.available) {
              <a
                class="flex h-full flex-col gap-2 rounded-lg border bg-card p-4 transition-colors hover:border-foreground/30 hover:bg-muted/40 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring motion-reduce:transition-none"
                [routerLink]="['/learn/algorithms', algorithm.id]"
                [attr.data-testid]="'algorithm-' + algorithm.id"
              >
                <span class="font-medium">{{ algorithm.name }}</span>
                @if (algorithm.summary) {
                  <span class="text-sm text-muted-foreground">{{ algorithm.summary }}</span>
                }
                <span class="mt-auto flex flex-wrap items-center gap-2 pt-1">
                  @for (task of algorithm.task_types; track task) {
                    <span hlmBadge variant="secondary">{{ taskLabels[task] }}</span>
                  }
                  @if (algorithm.difficulty) {
                    <span class="text-xs text-muted-foreground">
                      Difficulty {{ algorithm.difficulty }}/5
                    </span>
                  }
                </span>
              </a>
            } @else {
              <div
                class="flex h-full items-center justify-between gap-2 rounded-lg border border-dashed p-4 text-muted-foreground"
                [attr.data-testid]="'algorithm-' + algorithm.id"
              >
                <span>{{ algorithm.name }}</span>
                <span hlmBadge variant="outline" class="text-muted-foreground">Coming soon</span>
              </div>
            }
          </li>
        }
      </ul>
    </section>
  `,
})
export class FamilySection {
  readonly family = input.required<FamilyOut>();
  protected readonly taskLabels = TASK_LABELS;
}
