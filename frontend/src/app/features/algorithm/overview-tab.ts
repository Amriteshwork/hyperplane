import { ChangeDetectionStrategy, Component, inject } from '@angular/core';

import { MarkdownView } from '../../shared/ui/markdown-view/markdown-view';
import { AlgorithmStore } from './algorithm.store';

@Component({
  selector: 'app-overview-tab',
  imports: [MarkdownView],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    @if (store.algorithm(); as algorithm) {
      <app-markdown-view [content]="algorithm.overview" />
    }
  `,
})
export class OverviewTab {
  protected readonly store = inject(AlgorithmStore);
}
