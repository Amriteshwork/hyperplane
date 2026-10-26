import { ChangeDetectionStrategy, Component, effect, inject, input } from '@angular/core';
import { Title } from '@angular/platform-browser';
import { RouterLink } from '@angular/router';
import { HlmButton } from '@spartan-ng/helm/button';

import { EmptyState } from '../../shared/ui/empty-state/empty-state';
import { MarkdownView } from '../../shared/ui/markdown-view/markdown-view';
import { ResourceList } from '../../shared/ui/resource-list/resource-list';
import { TrackStore } from './track.store';

@Component({
  selector: 'app-track-page',
  imports: [RouterLink, HlmButton, EmptyState, MarkdownView, ResourceList],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="mx-auto max-w-6xl px-4 py-8 sm:py-12">
      @if (store.track(); as track) {
        <nav aria-label="Breadcrumb" class="text-sm text-muted-foreground">
          <ol class="flex items-center gap-1.5">
            <li><a routerLink="/learn" class="hover:text-foreground hover:underline">Learn</a></li>
          </ol>
        </nav>
        <p class="mt-3 text-xs font-medium tracking-wide text-muted-foreground uppercase">
          Guided track
        </p>
        <h1 class="mt-1 text-3xl font-semibold tracking-tight">{{ track.title }}</h1>
        <p class="mt-2 max-w-prose text-muted-foreground">{{ track.summary }}</p>
        <p class="mt-3 text-sm text-muted-foreground">
          {{ store.chapters().length }} chapters · {{ store.conceptCount() }} concepts · ends at
          <a class="underline hover:text-foreground" [routerLink]="['/concepts', track.goal.id]">
            {{ track.goal.title }}
          </a>
        </p>
        @if (store.firstConcept(); as first) {
          <a hlmBtn class="mt-4" [routerLink]="['/concepts', first.id]">
            Start with {{ first.title }} →
          </a>
        }

        <app-markdown-view class="mt-8 block max-w-3xl" [content]="track.body" />

        <section class="mt-12 max-w-3xl" aria-labelledby="route">
          <h2 id="route" class="text-xl font-semibold tracking-tight">The route</h2>
          <p class="mt-1 text-sm text-muted-foreground">
            Every concept comes after everything it builds on. Skip what you know.
          </p>
          <ol class="mt-6 grid gap-10">
            @for (chapter of store.chapters(); track chapter.title; let c = $index) {
              <li [attr.aria-labelledby]="'chapter-' + c">
                <p class="text-xs font-medium text-muted-foreground">Chapter {{ c + 1 }}</p>
                <h3 class="text-lg font-semibold" [id]="'chapter-' + c">{{ chapter.title }}</h3>
                <p class="mt-1 text-sm text-muted-foreground">{{ chapter.summary }}</p>
                <ol class="mt-3 grid gap-2 border-l-2 border-border pl-4">
                  @for (concept of chapter.concepts; track concept.id; let i = $index) {
                    <li>
                      <a
                        class="flex gap-3 rounded-lg border bg-card p-3 transition-colors hover:border-foreground/30 hover:bg-muted/40 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring motion-reduce:transition-none"
                        [class.border-primary]="concept.id === track.goal.id"
                        [routerLink]="['/concepts', concept.id]"
                      >
                        <span
                          class="flex size-6 shrink-0 items-center justify-center rounded-full bg-muted text-xs tabular-nums"
                          [class.bg-primary]="concept.id === track.goal.id"
                          [class.text-primary-foreground]="concept.id === track.goal.id"
                          aria-hidden="true"
                        >
                          {{ chapter.start + i }}
                        </span>
                        <span class="flex min-w-0 flex-col gap-0.5">
                          <span class="font-medium">{{ concept.title }}</span>
                          <span class="text-sm text-muted-foreground">{{ concept.summary }}</span>
                        </span>
                      </a>
                    </li>
                  }
                </ol>
              </li>
            }
          </ol>
        </section>

        <section class="mt-12" aria-labelledby="resources">
          <h2 id="resources" class="mb-3 text-lg font-semibold">Resources for the whole track</h2>
          <app-resource-list [resources]="store.resources()" />
        </section>

        @if (store.papers().length) {
          <section class="mt-10" aria-labelledby="papers">
            <h2 id="papers" class="mb-3 text-lg font-semibold">Research papers</h2>
            <app-resource-list [resources]="store.papers()" />
          </section>
        }
      } @else if (store.notFound()) {
        <app-empty-state heading="Track not found">
          <a hlmBtn variant="outline" routerLink="/concepts">Browse concepts</a>
        </app-empty-state>
      } @else if (store.error(); as error) {
        <app-empty-state heading="Couldn’t load this track" [description]="error">
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
export class TrackPage {
  protected readonly store = inject(TrackStore);
  private readonly title = inject(Title);

  readonly trackId = input.required<string>();

  constructor() {
    effect(() => this.store.trackId.set(this.trackId()));
    effect(() => {
      const title = this.store.track()?.title;
      if (title) this.title.setTitle(`${title} · Hyperplane`);
    });
  }
}
