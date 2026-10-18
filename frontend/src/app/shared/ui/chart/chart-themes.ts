import type { ThemeOption } from 'ngx-echarts';

/**
 * Colours validated with the dataviz palette checker against our card surfaces
 * (#ffffff light, #171717 dark). Categorical order is the CVD-safety mechanism: don't reorder.
 */
const PALETTE = {
  light: ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948'],
  dark: ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'],
};

const INK = {
  light: {
    text: '#171717',
    muted: '#737373',
    grid: '#e5e5e5',
    axis: '#d4d4d4',
    tooltip: '#ffffff',
  },
  dark: { text: '#fafafa', muted: '#a3a3a3', grid: '#2a2a2a', axis: '#404040', tooltip: '#262626' },
};

/** One hue, light → dark: magnitude (e.g. histogram density). */
export const SEQUENTIAL_RAMP = ['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'];

/** Blue ↔ grey ↔ red: signed values such as correlations or residuals. */
export const DIVERGING_RAMP = {
  light: ['#1c5cab', '#86b6ef', '#f0efec', '#f19a9a', '#b42828'],
  dark: ['#3987e5', '#1c5cab', '#383835', '#a33a3a', '#e66767'],
};

export type ChartMode = 'light' | 'dark';

export function chartTheme(mode: ChartMode): ThemeOption {
  const ink = INK[mode];
  const axis = {
    axisLine: { lineStyle: { color: ink.axis } },
    axisTick: { lineStyle: { color: ink.axis } },
    axisLabel: { color: ink.muted },
    splitLine: { lineStyle: { color: ink.grid } },
    nameTextStyle: { color: ink.muted },
  };
  return {
    color: PALETTE[mode],
    backgroundColor: 'transparent',
    textStyle: { color: ink.text, fontFamily: 'Inter Variable, ui-sans-serif, system-ui' },
    title: { textStyle: { color: ink.text }, subtextStyle: { color: ink.muted } },
    legend: { textStyle: { color: ink.muted } },
    tooltip: {
      backgroundColor: ink.tooltip,
      borderColor: ink.axis,
      textStyle: { color: ink.text },
    },
    valueAxis: axis,
    categoryAxis: { ...axis, splitLine: { show: false } },
    visualMap: { textStyle: { color: ink.muted } },
  };
}
