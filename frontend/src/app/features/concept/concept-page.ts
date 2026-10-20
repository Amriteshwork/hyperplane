import { ChangeDetectionStrategy, Component, effect, inject, input } from '@angular/core';
import { Title } from '@angular/platform-browser';
import { RouterLink } from '@angular/router';
import { HlmButton } from '@spartan-ng/helm/button';

import { EmptyState } from '../../shared/ui/empty-state/empty-state';
import { MarkdownView } from '../../shared/ui/markdown-view/markdown-view';
import { ResourceList } from '../../shared/ui/resource-list/resource-list';
import { ConceptStore } from './concept.store';

@Component({
  selector: 'app-concept-page',
  imports: [RouterLink, HlmButton, EmptyState, MarkdownView, ResourceList],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="mx-auto max-w-4xl px-4 py-8 sm:py-12">
      @if (store.concept(); as concept) {
        <nav aria-label="Breadcrumb" class="text-sm text-muted-foreground">
          <ol class="flex items-center gap-1.5">
            <li><a routerLink="/learn" class="hover:text-foreground hover:underline">Learn</a></li>
            <li aria-hidden="true">/</li>
            <li>Concepts</li>
          </ol>
        </nav>
        <h1 class="mt-3 text-3xl font-semibold tracking-tight">{{ concept.title }}</h1>
        <p class="mt-2 max-w-prose text-muted-foreground">{{ concept.summary }}</p>

        @if (concept.prerequisites.length) {
          <div class="mt-4 flex flex-wrap items-center gap-2 text-sm">
            <span class="text-muted-foreground">Read first:</span>
            @for (prerequisite of concept.prerequisites; track prerequisite.id) {
              <a
                class="rounded-full border px-2.5 py-0.5 text-xs hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring"
                [routerLink]="['/concepts', prerequisite.id]"
                >{{ prerequisite.title }}</a
              >
            }
          </div>
        }

        <app-markdown-view class="mt-8 block" [content]="concept.body" />

        @if (concept.algorithms.length) {
          <section class="mt-10" aria-labelledby="used-by">
            <h2 id="used-by" class="text-lg font-semibold">Used by</h2>
            <ul class="mt-3 flex flex-wrap gap-2">
              @for (algorithm of concept.algorithms; track algorithm.id) {
                <li>
                  <a
                    hlmBtn
                    variant="outline"
                    size="sm"
                    [routerLink]="['/learn/algorithms', algorithm.id]"
                  >
                    {{ algorithm.name }}
                  </a>
                </li>
              }
            </ul>
          </section>
        }

        <section class="mt-10" aria-labelledby="resources">
          <h2 id="resources" class="mb-3 text-lg font-semibold">Resources</h2>
          <app-resource-list [resources]="store.resources()" />
        </section>

        @if (store.papers().length) {
          <section class="mt-10" aria-labelledby="papers">
            <h2 id="papers" class="mb-3 text-lg font-semibold">Research papers</h2>
            <app-resource-list [resources]="store.papers()" />
          </section>
        }
      } @else if (store.notFound()) {
        <app-empty-state heading="Concept not found">
          <a hlmBtn variant="outline" routerLink="/learn">Back to Learn</a>
        </app-empty-state>
      } @else if (store.error(); as error) {
        <app-empty-state heading="Couldn’t load this concept" [description]="error">
          <button hlmBtn variant="outline" type="button" (click)="store.reload()">Try again</button>
        </app-empty-state>
      } @else {
        <div class="grid gap-3" aria-busy="true">
          <div class="h-9 w-64 animate-pulse rounded bg-muted motion-reduce:animate-none"></div>
          <div class="h-40 animate-pulse rounded bg-muted motion-reduce:animate-none"></div>
          <span class="sr-only">Loading…</span>
        </div>
      }
    </div>
  `,
})
export class ConceptPage {
  protected readonly store = inject(ConceptStore);
  private readonly title = inject(Title);

  readonly conceptId = input.required<string>();

  constructor() {
    effect(() => this.store.conceptId.set(this.conceptId()));
    effect(() => {
      const title = this.store.concept()?.title;
      if (title) this.title.setTitle(`${title} · Hyperplane`);
    });
  }
}
