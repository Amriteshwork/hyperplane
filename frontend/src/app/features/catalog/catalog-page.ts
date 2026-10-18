import { ChangeDetectionStrategy, Component } from '@angular/core';

import { ComingSoon } from '../../shared/ui/coming-soon/coming-soon';

@Component({
  selector: 'app-catalog-page',
  imports: [ComingSoon],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `<app-coming-soon
    heading="Learn"
    description="Browse algorithm families and pick something to study, from linear regression to gradient boosting."
    milestone="M0 · catalog, algorithm and concept pages"
  />`,
})
export class CatalogPage {}
