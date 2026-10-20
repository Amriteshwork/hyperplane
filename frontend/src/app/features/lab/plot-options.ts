import type { EChartsCoreOption } from 'echarts/core';

import { ScatterPlot } from '../../core/api/models/scatter-plot';

export type Plot = ScatterPlot;
export type PlotKind = NonNullable<Plot['kind']>;
type PlotOptionsBuilder = (plot: Plot) => EChartsCoreOption;

/** ECharts options per plot kind; a new evaluation plot = a new entry here. */
export const PLOT_OPTIONS: Record<PlotKind, PlotOptionsBuilder> = {
  scatter: scatterOptions,
};

export function plotOptions(plot: Plot): EChartsCoreOption {
  return PLOT_OPTIONS[plot.kind ?? 'scatter'](plot);
}

function scatterOptions(plot: ScatterPlot): EChartsCoreOption {
  return {
    grid: { left: 56, right: 16, top: 16, bottom: 48 },
    tooltip: { trigger: 'item' },
    xAxis: axis(plot.x_label),
    yAxis: axis(plot.y_label),
    series: [
      {
        type: 'scatter',
        symbolSize: 5,
        data: plot.points,
        markLine: plot.reference
          ? { silent: true, symbol: 'none', lineStyle: { type: 'dashed' }, data: guide(plot) }
          : undefined,
      },
    ],
  };
}

function axis(name: string): object {
  return { type: 'value', name, nameLocation: 'middle', nameGap: 32, scale: true };
}

/** `identity`: the y = x diagonal across the data; `zero`: the y = 0 line. */
function guide(plot: ScatterPlot): unknown[] {
  if (plot.reference === 'zero') return [{ yAxis: 0 }];
  const values = plot.points.flat();
  const low = Math.min(...values);
  const high = Math.max(...values);
  return [[{ coord: [low, low] }, { coord: [high, high] }]];
}
