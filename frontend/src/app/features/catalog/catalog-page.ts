import { ChangeDetectionStrategy, Component, inject } from '@angular/core';
import { HlmButton } from '@spartan-ng/helm/button';

import { EmptyState } from '../../shared/ui/empty-state/empty-state';
import { CatalogStore } from './catalog.store';
import { FamilySection } from './family-section';

@Component({
  selector: 'app-catalog-page',
  imports: [FamilySection, EmptyState, HlmButton],
  providers: [CatalogStore],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="mx-auto max-w-6xl px-4 py-10 sm:py-14">
      <h1 class="text-3xl font-semibold tracking-tight">Learn</h1>
      <p class="mt-3 max-w-prose text-muted-foreground">
        Algorithms grouped by family, from the simplest intuition to the most complex. Each page has
        the theory, the key concepts, and the exact code, explained line by line.
      </p>

      <div class="mt-10 grid gap-12" aria-live="polite" [attr.aria-busy]="store.loading()">
        @if (store.error(); as error) {
          <app-empty-state heading="Couldn’t load the catalog" [description]="error">
            <button hlmBtn variant="outline" type="button" (click)="store.reload()">
              Try again
            </button>
          </app-empty-state>
        } @else if (store.loading() && !store.families().length) {
          @for (_ of [1, 2]; track $index) {
            <div class="grid gap-3" aria-hidden="true">
              <div class="h-6 w-48 animate-pulse rounded bg-muted motion-reduce:animate-none"></div>
              <div class="grid gap-3 sm:grid-cols-3">
                @for (_ of [1, 2, 3]; track $index) {
                  <div
                    class="h-28 animate-pulse rounded-lg bg-muted motion-reduce:animate-none"
                  ></div>
                }
              </div>
            </div>
          }
          <span class="sr-only">Loading algorithms…</span>
        } @else {
          @for (family of store.families(); track family.id) {
            <app-family-section [family]="family" />
          }
        }
      </div>
    </div>
  `,
})
export class CatalogPage {
  protected readonly store = inject(CatalogStore);
}
