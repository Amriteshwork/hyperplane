import { ChangeDetectionStrategy, Component, inject } from '@angular/core';

import { ResourceList } from '../../shared/ui/resource-list/resource-list';
import { AlgorithmStore } from './algorithm.store';

@Component({
  selector: 'app-papers-tab',
  imports: [ResourceList],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <app-resource-list
      [resources]="store.papers()"
      emptyHeading="No research papers yet"
      emptyDescription="We list a paper only when it's freely available and genuinely helps understand this algorithm."
    />
  `,
})
export class PapersTab {
  protected readonly store = inject(AlgorithmStore);
}
