import { ChangeDetectionStrategy, Component, input } from '@angular/core';

import { EmptyState } from '../empty-state/empty-state';
import { LearningResource, ResourceCard } from '../resource-card/resource-card';

@Component({
  selector: 'app-resource-list',
  imports: [ResourceCard, EmptyState],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    @if (resources().length) {
      <ul class="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        @for (resource of resources(); track resource.url) {
          <li><app-resource-card class="h-full" [resource]="resource" /></li>
        }
      </ul>
    } @else {
      <app-empty-state [heading]="emptyHeading()" [description]="emptyDescription()" />
    }
  `,
})
export class ResourceList {
  readonly resources = input.required<readonly LearningResource[]>();
  readonly emptyHeading = input('Curated resources coming soon');
  readonly emptyDescription = input(
    "We only list material we've checked for accuracy, clarity and level. Nothing has passed review here yet.",
  );
}
