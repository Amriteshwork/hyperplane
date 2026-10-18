import { Provider } from '@angular/core';
import { provideEchartsCore } from 'ngx-echarts';

/** ECharts is loaded the first time a chart renders, keeping it out of the initial bundle. */
export function provideCharts(): Provider {
  return provideEchartsCore({ echarts: () => import('./echarts-setup').then((m) => m.echarts) });
}
