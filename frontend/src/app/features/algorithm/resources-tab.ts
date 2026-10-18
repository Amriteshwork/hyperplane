import { ChangeDetectionStrategy, Component, inject } from '@angular/core';

import { ResourceList } from '../../shared/ui/resource-list/resource-list';
import { AlgorithmStore } from './algorithm.store';

@Component({
  selector: 'app-resources-tab',
  imports: [ResourceList],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `<app-resource-list [resources]="store.resources()" />`,
})
export class ResourcesTab {
  protected readonly store = inject(AlgorithmStore);
}
