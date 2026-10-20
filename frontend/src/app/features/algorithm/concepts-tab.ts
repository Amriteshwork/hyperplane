import { ChangeDetectionStrategy, Component, computed, inject } from '@angular/core';
import { RouterLink } from '@angular/router';
import { NgIcon, provideIcons } from '@ng-icons/core';
import { lucideArrowRight } from '@ng-icons/lucide';
import { HlmBadge } from '@spartan-ng/helm/badge';

import { EmptyState } from '../../shared/ui/empty-state/empty-state';
import { AlgorithmStore } from './algorithm.store';

@Component({
  selector: 'app-concepts-tab',
  imports: [RouterLink, NgIcon, HlmBadge, EmptyState],
  providers: [provideIcons({ lucideArrowRight })],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    @if (store.algorithm(); as algorithm) {
      <p class="max-w-prose text-sm text-muted-foreground">
        Everything this algorithm builds on, down to the maths, in an order where each concept comes
        after what it needs. Skip what you already know.
      </p>
      @if (algorithm.learning_path.length) {
        <ol class="mt-4 grid gap-2 sm:grid-cols-2" aria-label="Learning path">
          @for (concept of algorithm.learning_path; track concept.id; let i = $index) {
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
                @if (keyConcepts().has(concept.id)) {
                  <span hlmBadge variant="secondary" data-testid="key-concept">Key concept</span>
                }
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

  /** The concepts the algorithm lists itself; the rest is background they build on. */
  protected readonly keyConcepts = computed(
    () => new Set(this.store.algorithm()?.concepts.map((c) => c.id)),
  );
}
