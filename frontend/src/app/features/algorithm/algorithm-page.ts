import { ChangeDetectionStrategy, Component, input } from '@angular/core';

import { ComingSoon } from '../../shared/ui/coming-soon/coming-soon';

@Component({
  selector: 'app-algorithm-page',
  imports: [ComingSoon],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `<app-coming-soon
    [heading]="algorithmId()"
    description="Overview, concepts, curated resources, the explained code, and a way to try it on data."
    milestone="M0 · catalog, algorithm and concept pages"
  />`,
})
export class AlgorithmPage {
  /** Bound from the route parameter. */
  readonly algorithmId = input.required<string>();
}
