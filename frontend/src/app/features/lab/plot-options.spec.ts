import { CurvePlot } from '../../core/api/models/curve-plot';
import { HeatmapPlot } from '../../core/api/models/heatmap-plot';
import { ScatterPlot } from '../../core/api/models/scatter-plot';
import { barOptions, plotOptions } from './plot-options';

const plot: ScatterPlot = {
  kind: 'scatter',
  id: 'p',
  title: 'T',
  description: 'D',
  x_label: 'Actual',
  y_label: 'Predicted',
  points: [
    [1, 4],
    [3, 2],
  ],
  reference: 'identity',
};

interface Series {
  data: unknown;
  markLine?: { data: unknown };
}

function series(options: object): Series {
  return (options as { series: Series[] }).series[0];
}

describe('plotOptions', () => {
  it('plots the points with a y = x guide spanning the data', () => {
    const options = plotOptions(plot);
    expect(series(options).data).toEqual(plot.points);
    expect(series(options).markLine?.data).toEqual([[{ coord: [1, 1] }, { coord: [4, 4] }]]);
  });

  it('draws a zero line for residual plots and none without a reference', () => {
    expect(series(plotOptions({ ...plot, reference: 'zero' })).markLine?.data).toEqual([
      { yAxis: 0 },
    ]);
    expect(series(plotOptions({ ...plot, reference: null })).markLine).toBeUndefined();
  });
});

describe('plotOptions for classification', () => {
  const confusion: HeatmapPlot = {
    kind: 'heatmap',
    id: 'confusion-matrix',
    title: 'Confusion matrix',
    description: 'D',
    x_label: 'Predicted',
    y_label: 'Actual',
    x_categories: ['a', 'b'],
    y_categories: ['a', 'b'],
    values: [
      [8, 1],
      [2, 9],
    ],
  };
  const roc: CurvePlot = {
    kind: 'curve',
    id: 'roc',
    title: 'ROC',
    description: 'D',
    x_label: 'FPR',
    y_label: 'TPR',
    series: [
      {
        name: 'a',
        points: [
          [0, 0],
          [0.2, 0.8],
          [1, 1],
        ],
      },
      {
        name: 'b',
        points: [
          [0, 0],
          [1, 1],
        ],
      },
    ],
    reference: 'diagonal',
  };

  it('lays a heatmap out as a table: [column, row, count] with rows top-down', () => {
    const options = plotOptions(confusion) as {
      yAxis: { inverse: boolean; data: string[] };
      visualMap: { max: number };
    };
    const cells = series(options).data as { value: number[]; label: { color: string } }[];
    expect(cells.map((c) => c.value)).toEqual([
      [0, 0, 8],
      [1, 0, 1],
      [0, 1, 2],
      [1, 1, 9],
    ]);
    expect(options.yAxis.inverse).toBe(true);
    expect(options.visualMap.max).toBe(9);
    expect(cells[0].label.color).not.toEqual(cells[1].label.color); // readable on dark cells
  });

  it('draws one line per curve, a legend for several, and the diagonal once', () => {
    const options = plotOptions(roc) as { series: Series[]; legend?: object };
    expect(options.series.map((s) => s.data)).toEqual(roc.series.map((s) => s.points));
    expect(options.legend).toBeDefined();
    expect(options.series[0].markLine?.data).toEqual([[{ coord: [0, 0] }, { coord: [1, 1] }]]);
    expect(options.series[1].markLine).toBeUndefined();
    expect(plotOptions({ ...roc, series: [roc.series[0]] })).not.toHaveProperty('legend.top');
  });

  it('fits the axes to the data for curves that are not rates, e.g. a loss per epoch', () => {
    const rates = plotOptions(roc) as {
      xAxis: { min?: number; scale: boolean };
      yAxis: { max?: number };
    };
    expect(rates.xAxis.min).toBe(0);
    expect(rates.yAxis.max).toBe(1);
    const loss = plotOptions({ ...roc, axes: 'data', reference: null }) as {
      xAxis: { min?: number; scale: boolean };
      yAxis: { max?: number };
    };
    expect(loss.xAxis.min).toBeUndefined();
    expect(loss.xAxis.scale).toBe(true);
    expect(loss.yAxis.max).toBeUndefined();
  });
});

describe('curve markers', () => {
  const loss: CurvePlot = {
    kind: 'curve',
    id: 'loss-curve',
    title: 'T',
    description: 'D',
    x_label: 'Epoch',
    y_label: 'Loss',
    series: [
      {
        name: 'Loss',
        points: [
          [1, 2],
          [2, 1],
        ],
      },
    ],
    axes: 'data',
  };

  it('draws a labelled vertical line at the marker', () => {
    const options = plotOptions({ ...loss, marker: { x: 2, label: 'Weights kept (epoch 2)' } });
    expect(series(options).markLine?.data).toEqual([
      {
        xAxis: 2,
        label: { formatter: 'Weights kept (epoch 2)', position: 'end' },
      },
    ]);
  });

  it('draws nothing without a marker or reference', () => {
    expect(series(plotOptions(loss)).markLine).toBeUndefined();
  });
});

describe('barOptions', () => {
  it('puts the first bar at the top', () => {
    const options = barOptions(
      [
        { label: 'bmi', value: 3 },
        { label: 'age', value: 1 },
      ],
      'Weight',
    ) as { yAxis: { data: string[] }; series: { data: number[] }[] };
    expect(options.yAxis.data).toEqual(['age', 'bmi']);
    expect(options.series[0].data).toEqual([1, 3]);
  });
});
