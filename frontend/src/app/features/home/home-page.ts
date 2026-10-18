import { ChangeDetectionStrategy, Component, inject } from '@angular/core';
import { RouterLink } from '@angular/router';
import { HlmBadge } from '@spartan-ng/helm/badge';
import { HlmButton } from '@spartan-ng/helm/button';
import { HlmCardImports } from '@spartan-ng/helm/card';

import { HealthStore } from './health.store';
import { ServiceStatusList } from './service-status-list';

@Component({
  selector: 'app-home-page',
  imports: [RouterLink, HlmBadge, HlmButton, HlmCardImports, ServiceStatusList],
  providers: [HealthStore],
  changeDetection: ChangeDetectionStrategy.OnPush,
  templateUrl: './home-page.html',
})
export class HomePage {
  protected readonly health = inject(HealthStore);
}
