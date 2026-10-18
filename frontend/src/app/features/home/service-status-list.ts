import { ChangeDetectionStrategy, Component, input } from '@angular/core';
import { HlmBadge } from '@spartan-ng/helm/badge';

import { ServiceRow } from './health.store';

@Component({
  selector: 'app-service-status-list',
  imports: [HlmBadge],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <ul class="divide-y">
      @for (row of rows(); track row.id) {
        <li
          class="flex items-center justify-between gap-4 py-3"
          [attr.data-testid]="'service-' + row.id"
        >
          <span class="font-medium">{{ row.label }}</span>
          <span class="flex items-center gap-3">
            @if (row.latencyMs !== null) {
              <span class="font-mono text-xs text-muted-foreground">{{ row.latencyMs }} ms</span>
            }
            @if (row.up) {
              <span hlmBadge class="bg-success/15 text-success">Up</span>
            } @else {
              <span hlmBadge variant="destructive" [attr.title]="row.error">Down</span>
            }
          </span>
        </li>
      }
    </ul>
  `,
})
export class ServiceStatusList {
  readonly rows = input.required<readonly ServiceRow[]>();
}
