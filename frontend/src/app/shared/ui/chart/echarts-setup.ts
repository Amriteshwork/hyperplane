import { BarChart, HeatmapChart, LineChart, ScatterChart } from 'echarts/charts';
import {
  AriaComponent,
  GridComponent,
  LegendComponent,
  MarkLineComponent,
  TitleComponent,
  TooltipComponent,
  VisualMapComponent,
} from 'echarts/components';
import * as echarts from 'echarts/core';
import { CanvasRenderer } from 'echarts/renderers';

// Only what our plots use; register a new chart type here when an evaluation view needs it.
echarts.use([
  BarChart,
  HeatmapChart,
  LineChart,
  ScatterChart,
  AriaComponent,
  GridComponent,
  LegendComponent,
  MarkLineComponent,
  TitleComponent,
  TooltipComponent,
  VisualMapComponent,
  CanvasRenderer,
]);

export { echarts };
