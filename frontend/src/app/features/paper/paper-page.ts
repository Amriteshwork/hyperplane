import {
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  effect,
  inject,
  input,
  signal,
  viewChildren,
} from '@angular/core';
import { Title } from '@angular/platform-browser';
import { RouterLink } from '@angular/router';
import { HlmButton } from '@spartan-ng/helm/button';

import { EmptyState } from '../../shared/ui/empty-state/empty-state';
import { MarkdownView } from '../../shared/ui/markdown-view/markdown-view';
import { ResourceList } from '../../shared/ui/resource-list/resource-list';
import { PaperStore } from './paper.store';

/** A paper or blog post explained section by section, with an outline that follows the reader. */
@Component({
  selector: 'app-paper-page',
  imports: [RouterLink, HlmButton, EmptyState, MarkdownView, ResourceList],
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: { '(window:scroll)': 'onScroll()' },
  template: `
    <div class="mx-auto max-w-6xl px-4 py-8 sm:py-12">
      @if (store.paper(); as paper) {
        <nav aria-label="Breadcrumb" class="text-sm text-muted-foreground">
          <ol class="flex items-center gap-1.5">
            <li>
              <a routerLink="/papers" class="hover:text-foreground hover:underline">Papers</a>
            </li>
          </ol>
        </nav>
        <p class="mt-3 text-xs font-medium tracking-wide text-muted-foreground uppercase">
          {{ paper.source.type === 'blog' ? 'Blog post' : 'Paper' }} walkthrough
        </p>
        <h1 class="mt-1 max-w-3xl text-3xl font-semibold tracking-tight">{{ paper.title }}</h1>
        <p class="mt-2 max-w-prose text-muted-foreground">{{ paper.summary }}</p>
        <p class="mt-3 text-sm text-muted-foreground">
          Original:
          <a
            class="underline underline-offset-4 hover:text-foreground"
            [href]="paper.source.url"
            target="_blank"
            rel="noopener noreferrer"
            >{{ paper.source.title }}<span class="sr-only"> (opens in a new tab)</span></a
          >
          @if (paper.source.author) {
            · {{ paper.source.author }}
          }
        </p>

        <aside
          class="mt-8 max-w-3xl rounded-lg border bg-card p-4 sm:p-5"
          aria-labelledby="before-you-read"
        >
          <h2 id="before-you-read" class="font-semibold">Before you read</h2>
          <p class="mt-1 text-sm text-muted-foreground">
            The walkthrough leans on these ideas. Each links to a page that builds it from scratch.
            Skip the ones you know.
          </p>
          <ul class="mt-3 flex flex-wrap gap-2">
            @for (concept of paper.concepts; track concept.id) {
              <li>
                <a
                  class="inline-block rounded-full border px-3 py-1 text-sm transition-colors hover:border-foreground/30 hover:bg-muted/40 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring motion-reduce:transition-none"
                  [routerLink]="['/concepts', concept.id]"
                  [title]="concept.summary"
                  >{{ concept.title }}</a
                >
              </li>
            }
          </ul>
          @if (paper.track; as track) {
            <p class="mt-3 text-sm">
              New to all of them?
              <a class="underline underline-offset-4" [routerLink]="['/tracks', track.id]">{{
                track.title
              }}</a>
              builds every one, in order.
            </p>
          }
        </aside>

        <app-markdown-view class="mt-8 block max-w-3xl" [content]="paper.intro" />

        <div class="mt-10 grid grid-cols-[minmax(0,1fr)] gap-10 lg:grid-cols-[13rem_minmax(0,1fr)]">
          <nav aria-labelledby="outline" class="lg:sticky lg:top-6 lg:self-start">
            <h2 id="outline" class="text-sm font-semibold">In this walkthrough</h2>
            <ol class="mt-2 grid gap-1 border-l text-sm">
              @for (section of store.sections(); track section.id) {
                <li>
                  <a
                    class="-ml-px block border-l-2 border-transparent py-1 pl-3 text-muted-foreground hover:text-foreground"
                    [class.border-primary]="active() === section.id"
                    [class.text-foreground]="active() === section.id"
                    [class.font-medium]="active() === section.id"
                    [attr.aria-current]="active() === section.id ? 'location' : null"
                    [routerLink]="[]"
                    [fragment]="section.id"
                    >{{ section.title }}</a
                  >
                </li>
              }
            </ol>
          </nav>

          <div class="grid min-w-0 grid-cols-[minmax(0,1fr)] gap-14">
            @for (section of store.sections(); track section.id) {
              <section #section [id]="section.id" [attr.aria-labelledby]="section.id + '-h'">
                <h2 [id]="section.id + '-h'" class="text-2xl font-semibold tracking-tight">
                  {{ section.title }}
                </h2>
                <app-markdown-view
                  class="mt-4 block"
                  [content]="section.body"
                  [videos]="store.videos()"
                />
              </section>
            }

            @if (store.resources().length) {
              <section aria-labelledby="resources">
                <h2 id="resources" class="mb-3 text-lg font-semibold">Read or watch next</h2>
                <app-resource-list [resources]="store.resources()" />
              </section>
            }
          </div>
        </div>
      } @else if (store.notFound()) {
        <app-empty-state heading="Walkthrough not found">
          <a hlmBtn variant="outline" routerLink="/papers">All walkthroughs</a>
        </app-empty-state>
      } @else if (store.error(); as error) {
        <app-empty-state heading="Couldn’t load this walkthrough" [description]="error">
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
export class PaperPage {
  protected readonly store = inject(PaperStore);
  private readonly title = inject(Title);
  private readonly sectionEls = viewChildren<ElementRef<HTMLElement>>('section');

  readonly paperId = input.required<string>();
  /** The section being read: the last one whose top has scrolled past a third of the screen. */
  protected readonly active = signal<string | null>(null);
  private frame = 0;

  constructor() {
    effect(() => this.store.paperId.set(this.paperId()));
    effect(() => {
      const title = this.store.paper()?.title;
      if (title) this.title.setTitle(`${title} · Hyperplane`);
    });
  }

  protected onScroll(): void {
    if (this.frame) return;
    this.frame = requestAnimationFrame(() => {
      this.frame = 0;
      const line = window.innerHeight / 3;
      const passed = this.sectionEls().filter(
        (el) => el.nativeElement.getBoundingClientRect().top <= line,
      );
      this.active.set(passed.at(-1)?.nativeElement.id ?? null);
    });
  }
}
