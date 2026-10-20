import { ChangeDetectionStrategy, Component, inject } from '@angular/core';
import { RouterLink } from '@angular/router';
import { HlmButton } from '@spartan-ng/helm/button';
import { HlmInput } from '@spartan-ng/helm/input';

import { EmptyState } from '../../shared/ui/empty-state/empty-state';
import { ConceptIndexStore } from './concept-index.store';

@Component({
  selector: 'app-concept-index-page',
  imports: [RouterLink, HlmButton, HlmInput, EmptyState],
  providers: [ConceptIndexStore],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="mx-auto max-w-6xl px-4 py-10 sm:py-14">
      <h1 class="text-3xl font-semibold tracking-tight">Concepts</h1>
      <p class="mt-3 max-w-prose text-muted-foreground">
        Every idea on Hyperplane, from NumPy and the maths up. Each page has the explanation, a
        from-scratch implementation that builds on earlier concepts, and the best links to learn
        more. Within each area, concepts come after what they need.
      </p>

      <div class="mt-6 max-w-md">
        <label for="concept-search" class="sr-only">Search concepts</label>
        <input
          id="concept-search"
          hlmInput
          type="search"
          class="w-full"
          placeholder="Search concepts, e.g. gradient"
          autocomplete="off"
          [value]="store.query()"
          (input)="onSearch($event)"
        />
        @if (store.query()) {
          <p class="mt-2 text-sm text-muted-foreground" role="status">
            {{ store.matches() }} of {{ store.total() }} concepts
          </p>
        }
      </div>

      <div class="mt-10 grid gap-12" [attr.aria-busy]="store.loading()">
        @if (store.error(); as error) {
          <app-empty-state heading="Couldn’t load the concepts" [description]="error">
            <button hlmBtn variant="outline" type="button" (click)="store.reload()">
              Try again
            </button>
          </app-empty-state>
        } @else if (store.loading() && !store.total()) {
          <div class="grid gap-3 sm:grid-cols-3" aria-hidden="true">
            @for (_ of [1, 2, 3]; track $index) {
              <div class="h-28 animate-pulse rounded-lg bg-muted motion-reduce:animate-none"></div>
            }
          </div>
          <span class="sr-only">Loading concepts…</span>
        } @else if (!store.filtered().length) {
          <app-empty-state
            heading="No concepts match"
            description="Try a shorter or different word."
          >
            <button hlmBtn variant="outline" type="button" (click)="store.query.set('')">
              Clear search
            </button>
          </app-empty-state>
        } @else {
          @for (area of store.filtered(); track area.id) {
            <section [attr.aria-labelledby]="'area-' + area.id">
              <h2 class="text-xl font-semibold tracking-tight" [id]="'area-' + area.id">
                {{ area.title }}
              </h2>
              <ul class="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                @for (concept of area.concepts; track concept.id) {
                  <li>
                    <a
                      class="flex h-full flex-col gap-2 rounded-lg border bg-card p-4 transition-colors hover:border-foreground/30 hover:bg-muted/40 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring motion-reduce:transition-none"
                      [routerLink]="['/concepts', concept.id]"
                    >
                      <span class="font-medium">{{ concept.title }}</span>
                      <span class="text-sm text-muted-foreground">{{ concept.summary }}</span>
                      @if (concept.algorithms.length) {
                        <span class="mt-auto pt-1 text-xs text-muted-foreground">
                          Used by
                          @for (
                            algorithm of concept.algorithms;
                            track algorithm.id;
                            let last = $last
                          ) {
                            {{ algorithm.name }}{{ last ? '' : ', ' }}
                          }
                        </span>
                      }
                    </a>
                  </li>
                }
              </ul>
            </section>
          }
        }
      </div>
    </div>
  `,
})
export class ConceptIndexPage {
  protected readonly store = inject(ConceptIndexStore);

  protected onSearch(event: Event): void {
    this.store.query.set((event.target as HTMLInputElement).value);
  }
}
