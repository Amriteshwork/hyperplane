import { ChangeDetectionStrategy, Component, inject } from '@angular/core';

import { MarkdownView } from '../../shared/ui/markdown-view/markdown-view';
import { ResourceCard } from '../../shared/ui/resource-card/resource-card';
import { AlgorithmStore } from './algorithm.store';

@Component({
  selector: 'app-overview-tab',
  imports: [MarkdownView, ResourceCard],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    @if (store.papers().length) {
      <section class="mb-8" aria-labelledby="papers-heading">
        <h2 id="papers-heading" class="mb-3 text-lg font-semibold">Research papers</h2>
        <ul class="grid gap-4 sm:grid-cols-2">
          @for (paper of store.papers(); track paper.url) {
            <li><app-resource-card class="h-full" [resource]="paper" /></li>
          }
        </ul>
      </section>
    }
    @if (store.algorithm(); as algorithm) {
      <app-markdown-view [content]="algorithm.overview" />
    }
  `,
})
export class OverviewTab {
  protected readonly store = inject(AlgorithmStore);
}
