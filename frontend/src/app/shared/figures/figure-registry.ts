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
  'rope-rotation': () => import('./rope-rotation-figure').then((m) => m.RopeRotationFigure),
  'rope-decay': () => import('./rope-decay-figure').then((m) => m.RopeDecayFigure),
  'rope-bands': () => import('./rope-bands-figure').then((m) => m.RopeBandsFigure),
  'yarn-frequencies': () =>
    import('./yarn-frequencies-figure').then((m) => m.YarnFrequenciesFigure),
  'transformer-block': () =>
    import('./transformer-block-figure').then((m) => m.TransformerBlockFigure),
  'bptt-unrolled': () => import('./bptt-unrolled-figure').then((m) => m.BpttUnrolledFigure),
  'fill-in-the-blank': () =>
    import('./fill-in-the-blank-figure').then((m) => m.FillInTheBlankFigure),
  'bert-input': () => import('./bert-input-figure').then((m) => m.BertInputFigure),
  'bert-heads': () => import('./bert-heads-figure').then((m) => m.BertHeadsFigure),
  'cross-attention': () => import('./cross-attention-figure').then((m) => m.CrossAttentionFigure),
  'seq2seq-steps': () => import('./seq2seq-steps-figure').then((m) => m.Seq2seqStepsFigure),
  'tokenizer-playground': () =>
    import('./tokenizer-playground-figure').then((m) => m.TokenizerPlaygroundFigure),
  'token-to-vector': () => import('./token-to-vector-figure').then((m) => m.TokenToVectorFigure),
  'sampling-explorer': () =>
    import('./sampling-explorer-figure').then((m) => m.SamplingExplorerFigure),
  'temperature-tradeoff': () =>
    import('./temperature-tradeoff-figure').then((m) => m.TemperatureTradeoffFigure),
  'beam-search-tree': () => import('./beam-search-tree-figure').then((m) => m.BeamSearchTreeFigure),
  'kv-cache': () => import('./kv-cache-figure').then((m) => m.KvCacheFigure),
  'bottleneck-curve': () =>
    import('./bottleneck-curve-figure').then((m) => m.BottleneckCurveFigure),
  'attention-lookback': () =>
    import('./attention-lookback-figure').then((m) => m.AttentionLookbackFigure),
  'skipgram-trainer': () =>
    import('./skipgram-trainer-figure').then((m) => m.SkipgramTrainerFigure),
  'word-map': () => import('./word-map-figure').then((m) => m.WordMapFigure),
  'lr-warmup': () => import('./lr-warmup-figure').then((m) => m.LrWarmupFigure),
  'gradient-clip': () => import('./gradient-clip-figure').then((m) => m.GradientClipFigure),
  'dropout-mask': () => import('./dropout-mask-figure').then((m) => m.DropoutMaskFigure),
  'chat-labels': () => import('./chat-labels-figure').then((m) => m.ChatLabelsFigure),
  'base-vs-chat': () => import('./base-vs-chat-figure').then((m) => m.BaseVsChatFigure),
  'reward-votes': () => import('./reward-votes-figure').then((m) => m.RewardVotesFigure),
  'rlhf-leash': () => import('./rlhf-leash-figure').then((m) => m.RlhfLeashFigure),
  'bayes-base-rate': () => import('./bayes-base-rate-figure').then((m) => m.BayesBaseRateFigure),
  'likelihood-explorer': () =>
    import('./likelihood-explorer-figure').then((m) => m.LikelihoodExplorerFigure),
  'double-counting': () => import('./double-counting-figure').then((m) => m.DoubleCountingFigure),
  'naive-bayes-playground': () =>
    import('./naive-bayes-playground-figure').then((m) => m.NaiveBayesPlaygroundFigure),
  'knn-playground': () => import('./knn-playground-figure').then((m) => m.KnnPlaygroundFigure),
  'knn-regression': () => import('./knn-regression-figure').then((m) => m.KnnRegressionFigure),
  'distance-balls': () => import('./distance-balls-figure').then((m) => m.DistanceBallsFigure),
  'distance-concentration': () =>
    import('./distance-concentration-figure').then((m) => m.DistanceConcentrationFigure),
};

export const FIGURES = new InjectionToken<Readonly<Record<string, FigureLoader>>>('FIGURES', {
  factory: () => DEFAULT_FIGURES,
});
