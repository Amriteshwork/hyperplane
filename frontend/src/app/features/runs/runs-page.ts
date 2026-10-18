import { ChangeDetectionStrategy, Component } from '@angular/core';

import { ComingSoon } from '../../shared/ui/coming-soon/coming-soon';

@Component({
  selector: 'app-runs-page',
  imports: [ComingSoon],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `<app-coming-soon
    heading="Runs"
    description="Your training runs: metrics, charts, code and downloadable models."
    milestone="M0 · worker + job pipeline"
  />`,
})
export class RunsPage {}
