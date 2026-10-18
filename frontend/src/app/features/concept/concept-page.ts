import { ChangeDetectionStrategy, Component, input } from '@angular/core';

import { ComingSoon } from '../../shared/ui/coming-soon/coming-soon';

@Component({
  selector: 'app-concept-page',
  imports: [ComingSoon],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `<app-coming-soon
    [heading]="conceptId()"
    description="A short explanation of one idea, with the best blogs and videos we could find."
    milestone="M0 · catalog, algorithm and concept pages"
  />`,
})
export class ConceptPage {
  /** Bound from the route parameter. */
  readonly conceptId = input.required<string>();
}
