import { ChangeDetectionStrategy, Component, inject } from '@angular/core';
import { RouterLink } from '@angular/router';
import { NgIcon, provideIcons } from '@ng-icons/core';
import { lucideArrowRight } from '@ng-icons/lucide';

import { EmptyState } from '../../shared/ui/empty-state/empty-state';
import { AlgorithmStore } from './algorithm.store';

@Component({
  selector: 'app-concepts-tab',
  imports: [RouterLink, NgIcon, EmptyState],
  providers: [provideIcons({ lucideArrowRight })],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    @if (store.algorithm(); as algorithm) {
      <p class="max-w-prose text-sm text-muted-foreground">
        The ideas this algorithm introduces, in a sensible reading order.
      </p>
      @if (algorithm.concepts.length) {
        <ol class="mt-4 grid gap-2 sm:grid-cols-2">
          @for (concept of algorithm.concepts; track concept.id; let i = $index) {
            <li>
              <a
                class="flex items-center gap-3 rounded-lg border p-3 hover:bg-muted/40 focus-visible:outline-2 focus-visible:outline-ring"
                [routerLink]="['/concepts', concept.id]"
              >
                <span
                  class="grid size-7 shrink-0 place-items-center rounded-full bg-muted text-xs font-medium"
                  aria-hidden="true"
                  >{{ i + 1 }}</span
                >
                <span class="flex-1 font-medium">{{ concept.title }}</span>
                <ng-icon name="lucideArrowRight" class="text-muted-foreground" aria-hidden="true" />
              </a>
            </li>
          }
        </ol>
      } @else {
        <app-empty-state class="mt-4" heading="No concepts listed yet" />
      }
    }
  `,
})
export class ConceptsTab {
  protected readonly store = inject(AlgorithmStore);
}
