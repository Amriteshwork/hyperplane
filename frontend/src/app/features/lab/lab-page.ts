import { ChangeDetectionStrategy, Component } from '@angular/core';

import { ComingSoon } from '../../shared/ui/coming-soon/coming-soon';

@Component({
  selector: 'app-lab-page',
  imports: [ComingSoon],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `<app-coming-soon
    heading="Lab"
    description="Upload data, get recommendations, train a model and read the exact code that ran."
    milestone="M0 · Lab flow end-to-end"
  />`,
})
export class LabPage {}
