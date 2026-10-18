import { ChangeDetectionStrategy, Component, effect, inject, input } from '@angular/core';
import { Title } from '@angular/platform-browser';
import { RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';
import { NgIcon, provideIcons } from '@ng-icons/core';
import { lucideFlaskConical } from '@ng-icons/lucide';
import { HlmBadge } from '@spartan-ng/helm/badge';
import { HlmButton } from '@spartan-ng/helm/button';

import { TASK_LABELS } from '../../shared/learn/labels';
import { EmptyState } from '../../shared/ui/empty-state/empty-state';
import { AlgorithmStore } from './algorithm.store';

const TABS = [
  { path: 'overview', label: 'Overview' },
  { path: 'concepts', label: 'Concepts' },
  { path: 'resources', label: 'Resources' },
  { path: 'code', label: 'Code' },
] as const;

/** Header + section navigation; each tab is a child route so it can be linked to. */
@Component({
  selector: 'app-algorithm-shell',
  imports: [RouterLink, RouterLinkActive, RouterOutlet, NgIcon, HlmBadge, HlmButton, EmptyState],
  providers: [provideIcons({ lucideFlaskConical })],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="mx-auto max-w-6xl px-4 py-8 sm:py-12">
      @if (store.algorithm(); as algorithm) {
        <nav aria-label="Breadcrumb" class="text-sm text-muted-foreground">
          <ol class="flex flex-wrap items-center gap-1.5">
            <li><a routerLink="/learn" class="hover:text-foreground hover:underline">Learn</a></li>
            @if (algorithm.family) {
              <li aria-hidden="true">/</li>
              <li>{{ algorithm.family.title }}</li>
            }
          </ol>
        </nav>
        <header class="mt-3 flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <h1 class="text-3xl font-semibold tracking-tight">{{ algorithm.name }}</h1>
            <p class="mt-2 max-w-prose text-muted-foreground">{{ algorithm.summary }}</p>
            <p class="mt-3 flex flex-wrap items-center gap-2">
              @for (task of algorithm.task_types; track task) {
                <span hlmBadge variant="secondary">{{ taskLabels[task] }}</span>
              }
              <span class="text-xs text-muted-foreground">
                Difficulty {{ algorithm.difficulty }}/5
              </span>
            </p>
          </div>
          <a
            hlmBtn
            class="shrink-0 self-start"
            routerLink="/lab"
            [queryParams]="{ algorithm: algorithm.id }"
            data-testid="try-it"
          >
            <ng-icon name="lucideFlaskConical" aria-hidden="true" />
            Try it in the Lab
          </a>
        </header>

        <nav aria-label="Algorithm sections" class="mt-8 border-b">
          <ul class="-mb-px flex gap-1 overflow-x-auto">
            @for (tab of tabs; track tab.path) {
              <li class="shrink-0">
                <a
                  class="inline-block border-b-2 border-transparent px-3 py-2 text-sm text-muted-foreground hover:text-foreground focus-visible:outline-2 focus-visible:outline-ring"
                  routerLinkActive="border-foreground! font-medium text-foreground!"
                  ariaCurrentWhenActive="page"
                  [routerLink]="tab.path"
                >
                  {{ tab.label }}
                </a>
              </li>
            }
          </ul>
        </nav>
        <div class="pt-8"><router-outlet /></div>
      } @else if (store.notFound()) {
        <app-empty-state
          heading="Algorithm not found"
          description="It may not have been added yet. Upcoming algorithms are listed, greyed out, in the catalog."
        >
          <a hlmBtn variant="outline" routerLink="/learn">Back to Learn</a>
        </app-empty-state>
      } @else if (store.error(); as error) {
        <app-empty-state heading="Couldn’t load this algorithm" [description]="error">
          <button hlmBtn variant="outline" type="button" (click)="store.reload()">Try again</button>
        </app-empty-state>
      } @else {
        <div class="grid gap-3" aria-busy="true">
          <div class="h-9 w-64 animate-pulse rounded bg-muted motion-reduce:animate-none"></div>
          <div
            class="h-5 w-96 max-w-full animate-pulse rounded bg-muted motion-reduce:animate-none"
          ></div>
          <span class="sr-only">Loading…</span>
        </div>
      }
    </div>
  `,
})
export class AlgorithmShell {
  protected readonly store = inject(AlgorithmStore);
  private readonly title = inject(Title);

  readonly algorithmId = input.required<string>();

  protected readonly tabs = TABS;
  protected readonly taskLabels = TASK_LABELS;

  constructor() {
    effect(() => this.store.algorithmId.set(this.algorithmId()));
    effect(() => {
      const name = this.store.algorithm()?.name;
      if (name) this.title.setTitle(`${name} · Hyperplane`);
    });
  }
}
