import { ScatterPlot } from '../../core/api/models/scatter-plot';
import { plotOptions } from './plot-options';

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
