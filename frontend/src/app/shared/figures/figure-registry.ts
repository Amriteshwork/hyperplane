import { InjectionToken, Type } from '@angular/core';

/** Loads a figure's component on demand, so pages only download the figures they show. */
export type FigureLoader = () => Promise<Type<unknown>>;

/**
 * Every figure content can embed with `::figure[name]`, keyed by that name. A new figure is a
 * new entry here; backend/tests/unit/test_content_figures.py checks content only uses these.
 */
export const DEFAULT_FIGURES: Readonly<Record<string, FigureLoader>> = {
  'xor-network': () => import('./xor-network-figure').then((m) => m.XorNetworkFigure),
  'network-shape': () => import('./network-shape-figure').then((m) => m.NetworkShapeFigure),
  'activation-explorer': () =>
    import('./activation-explorer-figure').then((m) => m.ActivationExplorerFigure),
  'relu-sum': () => import('./relu-sum-figure').then((m) => m.ReluSumFigure),
  'backprop-steps': () => import('./backprop-steps-figure').then((m) => m.BackpropStepsFigure),
  'optimizer-race': () => import('./optimizer-race-figure').then((m) => m.OptimizerRaceFigure),
  'early-stopping': () => import('./early-stopping-figure').then((m) => m.EarlyStoppingFigure),
  'boundary-playground': () =>
    import('./boundary-playground-figure').then((m) => m.BoundaryPlaygroundFigure),
  'attention-heatmap': () =>
    import('./attention-heatmap-figure').then((m) => m.AttentionHeatmapFigure),
  'positional-encoding-explorer': () =>
    import('./positional-encoding-explorer-figure').then((m) => m.PositionalEncodingExplorerFigure),
  'transformer-block': () =>
    import('./transformer-block-figure').then((m) => m.TransformerBlockFigure),
  'bptt-unrolled': () => import('./bptt-unrolled-figure').then((m) => m.BpttUnrolledFigure),
  'fill-in-the-blank': () =>
    import('./fill-in-the-blank-figure').then((m) => m.FillInTheBlankFigure),
  'bert-input': () => import('./bert-input-figure').then((m) => m.BertInputFigure),
  'bert-heads': () => import('./bert-heads-figure').then((m) => m.BertHeadsFigure),
};

export const FIGURES = new InjectionToken<Readonly<Record<string, FigureLoader>>>('FIGURES', {
  factory: () => DEFAULT_FIGURES,
});
