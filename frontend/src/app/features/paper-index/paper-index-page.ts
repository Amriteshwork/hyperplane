import { ChangeDetectionStrategy, Component, computed, inject, resource } from '@angular/core';
import { RouterLink } from '@angular/router';
import { HlmButton } from '@spartan-ng/helm/button';

import { Api } from '../../core/api/api';
import { listPapers } from '../../core/api/fn/learn/list-papers';
import { errorMessage } from '../../core/http/http-errors';
import { EmptyState } from '../../shared/ui/empty-state/empty-state';

/** Every paper and blog walkthrough, in authored order. */
@Component({
  selector: 'app-paper-index-page',
  imports: [RouterLink, HlmButton, EmptyState],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="mx-auto max-w-6xl px-4 py-10 sm:py-14">
      <h1 class="text-3xl font-semibold tracking-tight">Papers</h1>
      <p class="mt-3 max-w-prose text-muted-foreground">
        Landmark papers and blog posts, explained section by section in plain words. Every equation
        is unpacked, the key ideas are animated, and each section links to the concept pages that
        build its ideas from scratch.
      </p>

      <div class="mt-10" [attr.aria-busy]="page.isLoading()">
        @if (error(); as error) {
          <app-empty-state heading="Couldn’t load the papers" [description]="error">
            <button hlmBtn variant="outline" type="button" (click)="page.reload()">
              Try again
            </button>
          </app-empty-state>
        } @else if (page.isLoading() && !papers().length) {
          <div class="h-36 animate-pulse rounded-lg bg-muted motion-reduce:animate-none"></div>
          <span class="sr-only">Loading papers…</span>
        } @else if (!papers().length) {
          <app-empty-state heading="No walkthroughs yet" />
        } @else {
          <ul class="grid gap-4 sm:grid-cols-2">
            @for (paper of papers(); track paper.id) {
              <li>
                <a
                  class="flex h-full flex-col gap-2 rounded-lg border bg-card p-5 transition-colors hover:border-foreground/30 hover:bg-muted/40 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring motion-reduce:transition-none"
                  [routerLink]="['/papers', paper.id]"
                >
                  <span class="text-xs font-medium tracking-wide text-muted-foreground uppercase">
                    {{ paper.source.type === 'blog' ? 'Blog post' : 'Paper' }}
                    @if (paper.source.author) {
                      · {{ paper.source.author }}
                    }
                  </span>
                  <span class="text-lg font-semibold">{{ paper.title }}</span>
                  <span class="text-sm text-muted-foreground">{{ paper.summary }}</span>
                  <span class="mt-auto pt-1 text-xs text-muted-foreground">
                    {{ paper.section_count }} sections
                  </span>
                </a>
              </li>
            }
          </ul>
        }
      </div>
    </div>
  `,
})
export class PaperIndexPage {
  private readonly api = inject(Api);

  protected readonly page = resource({ loader: () => this.api.invoke(listPapers, { limit: 100 }) });
  protected readonly papers = computed(() => (this.page.hasValue() ? this.page.value().items : []));
  protected readonly error = computed(() =>
    this.page.error() ? errorMessage(this.page.error()) : null,
  );
}
