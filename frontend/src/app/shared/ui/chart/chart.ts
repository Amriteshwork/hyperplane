import {
  ChangeDetectionStrategy,
  Component,
  DOCUMENT,
  computed,
  inject,
  input,
} from '@angular/core';
import type { EChartsCoreOption } from 'echarts/core';
import { NgxEchartsDirective } from 'ngx-echarts';

import { chartTheme } from './chart-themes';

/**
 * Theme-aware ECharts figure. Callers pass `dark` (presentational: no theme service here) and an
 * accessible `label`; project a table or summary as the non-visual alternative.
 */
@Component({
  selector: 'app-chart',
  imports: [NgxEchartsDirective],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <figure class="grid gap-2">
      @if (caption()) {
        <figcaption class="text-sm font-medium">{{ caption() }}</figcaption>
      }
      <div
        echarts
        role="img"
        class="w-full"
        [attr.aria-label]="label()"
        [style.height]="height()"
        [options]="resolvedOptions()"
        [theme]="theme()"
      ></div>
      <ng-content />
    </figure>
  `,
})
export class Chart {
  private readonly reducedMotion =
    inject(DOCUMENT).defaultView?.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;

  readonly options = input.required<EChartsCoreOption>();
  readonly label = input.required<string>();
  readonly caption = input<string>();
  readonly dark = input(false);
  readonly height = input('320px');

  protected readonly theme = computed(() => chartTheme(this.dark() ? 'dark' : 'light'));
  protected readonly resolvedOptions = computed<EChartsCoreOption>(() => ({
    animation: !this.reducedMotion,
    ...this.options(),
  }));
}
