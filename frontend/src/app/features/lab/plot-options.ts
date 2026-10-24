import type { EChartsCoreOption } from 'echarts/core';

import { CurvePlot } from '../../core/api/models/curve-plot';
import { HeatmapPlot } from '../../core/api/models/heatmap-plot';
import { ScatterPlot } from '../../core/api/models/scatter-plot';
import { SEQUENTIAL_RAMP } from '../../shared/ui/chart/chart-themes';

export type Plot = ScatterPlot | HeatmapPlot | CurvePlot;
export type PlotKind = NonNullable<Plot['kind']>;
type PlotOf<K extends PlotKind> = Extract<Plot, { kind?: K }>;

/** ECharts options per plot kind; a new evaluation plot = a new entry here. */
export const PLOT_OPTIONS: { [K in PlotKind]: (plot: PlotOf<K>) => EChartsCoreOption } = {
  scatter: scatterOptions,
  heatmap: heatmapOptions,
  curve: curveOptions,
};

export function plotOptions(plot: Plot): EChartsCoreOption {
  // TypeScript can't pair a union member with its builder, but `kind` guarantees the match.
  const build = PLOT_OPTIONS[plot.kind ?? 'scatter'] as (plot: Plot) => EChartsCoreOption;
  return build(plot);
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

/** A labelled grid, first row at the top like a table (e.g. a confusion matrix). */
function heatmapOptions(plot: HeatmapPlot): EChartsCoreOption {
  const max = Math.max(1, ...plot.values.flat());
  const data = plot.values.flatMap((row, y) =>
    row.map((value, x) => ({
      value: [x, y, value],
      // The ramp darkens with the count: switch to light text on the darker half.
      label: { color: value > max / 2 ? '#ffffff' : '#171717' },
    })),
  );
  const categories = (name: string, data: string[]) => ({
    type: 'category',
    data,
    name,
    nameLocation: 'middle',
    nameGap: 28,
    axisLabel: { interval: 0 },
    splitArea: { show: false },
  });
  return {
    grid: { left: 16, right: 16, top: 16, bottom: 40, containLabel: true },
    tooltip: { position: 'top' },
    xAxis: categories(plot.x_label, plot.x_categories),
    yAxis: {
      ...categories(plot.y_label, plot.y_categories),
      inverse: true,
      nameGap: 0,
      nameLocation: 'end',
    },
    visualMap: { show: false, min: 0, max, inRange: { color: SEQUENTIAL_RAMP } },
    series: [{ type: 'heatmap', data, label: { show: true } }],
  };
}

/** Lines on a 0–1 square (e.g. ROC curves) or on axes fit to the data (e.g. a loss per epoch). */
function curveOptions(plot: CurvePlot): EChartsCoreOption {
  const several = plot.series.length > 1;
  const range = (name: string) =>
    plot.axes === 'data' ? axis(name) : { ...axis(name), scale: false, min: 0, max: 1 };
  return {
    grid: { left: 56, right: 16, top: several ? 40 : 16, bottom: 48 },
    tooltip: { trigger: 'item' },
    legend: several ? { top: 0 } : undefined,
    xAxis: range(plot.x_label),
    yAxis: range(plot.y_label),
    series: plot.series.map((line, i) => ({
      type: 'line',
      name: line.name,
      data: line.points,
      showSymbol: false,
      markLine:
        i === 0 && plot.reference === 'diagonal'
          ? {
              silent: true,
              symbol: 'none',
              lineStyle: { type: 'dashed' },
              data: [[{ coord: [0, 0] }, { coord: [1, 1] }]],
            }
          : undefined,
    })),
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
