import { ChangeDetectionStrategy, Component, effect, inject, input } from '@angular/core';
import { RouterLink, RouterLinkActive } from '@angular/router';

import { CodeExplainer } from '../../shared/ui/code-explainer/code-explainer';
import { EmptyState } from '../../shared/ui/empty-state/empty-state';
import { AlgorithmStore } from './algorithm.store';

@Component({
  selector: 'app-code-tab',
  imports: [RouterLink, RouterLinkActive, CodeExplainer, EmptyState],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    @if (store.algorithm(); as algorithm) {
      <div class="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <nav aria-label="Code variants">
          <ul class="inline-flex rounded-md border p-0.5">
            @for (option of algorithm.code_variants; track option.id) {
              <li>
                <a
                  class="block rounded px-3 py-1.5 text-sm text-muted-foreground hover:text-foreground focus-visible:outline-2 focus-visible:outline-ring"
                  routerLinkActive="bg-muted font-medium text-foreground!"
                  ariaCurrentWhenActive="page"
                  [routerLink]="['..', option.id]"
                >
                  {{ option.title }}
                </a>
              </li>
            }
          </ul>
        </nav>
        @if (store.code()?.source === 'example') {
          <p class="text-xs text-muted-foreground sm:max-w-sm sm:text-right">
            Shown with example settings. In the Lab, this script is generated from your data and
            choices, and it is the exact code that trains your model.
          </p>
        }
      </div>

      <div class="mt-4">
        @if (store.codeError(); as error) {
          <app-empty-state heading="Couldn’t load this code" [description]="error" />
        } @else if (store.code(); as code) {
          <app-code-explainer
            [code]="code.code"
            [annotations]="code.explained"
            [filename]="code.filename"
          />
        } @else {
          <div
            class="h-96 animate-pulse rounded-lg bg-muted motion-reduce:animate-none"
            aria-busy="true"
          ></div>
        }
      </div>
    }
  `,
})
export class CodeTab {
  protected readonly store = inject(AlgorithmStore);

  readonly variant = input.required<string>();

  constructor() {
    effect(() => this.store.variant.set(this.variant()));
  }
}
