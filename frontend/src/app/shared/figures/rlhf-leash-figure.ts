import { ChangeDetectionStrategy, Component, computed, signal } from '@angular/core';

import { FigureFrame } from './figure-frame';
import { FigureSlider } from './figure-slider';
import { linear, path, range } from './lib/plot';
import {
  BASE,
  LEARNED_WEIGHTS,
  REPLIES,
  TRUE_QUALITY,
  bestPolicy,
  expected,
  rewards,
} from './lib/rlhf';

const W = 320;
const H = 140;
const SCORES = rewards(LEARNED_WEIGHTS);
/** The slider is log β, so that loosening the leash moves left to right. */
const LOG_BETAS = range(Math.log10(50), Math.log10(0.2), 80);
const x = linear(Math.log10(50), Math.log10(0.2), 32, W - 8);
const qy = linear(-3.2, 3.5, H - 22, 8);
const ry = linear(0, 10, H - 22, 8);
const policyAt = (logBeta: number) => bestPolicy(BASE, SCORES, 10 ** logBeta);
const qualityLine = path(
  LOG_BETAS.map((b): [number, number] => [x(b), qy(expected(policyAt(b), TRUE_QUALITY))]),
);
const rewardLine = path(
  LOG_BETAS.map((b): [number, number] => [x(b), ry(expected(policyAt(b), SCORES))]),
);

/** Loosen the leash: the score keeps climbing while real quality rises, peaks, and collapses. */
@Component({
  selector: 'app-rlhf-leash-figure',
  imports: [FigureFrame, FigureSlider],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <app-figure-frame
      heading="Goodhart's law: the reward keeps rising, quality doesn't"
      hint="Slide right to loosen the leash, so the model chases the reward harder."
    >
      <app-figure-slider
        label="Leash: tight → loose"
        [(value)]="looseness"
        [min]="0"
        [max]="100"
        [readout]="'β = ' + beta().toFixed(beta() < 1 ? 2 : 1)"
      />

      <ul class="list-none pl-0 mt-3 space-y-1 text-xs" role="img" [attr.aria-label]="summary()">
        @for (row of rows(); track row.reply) {
          <li class="grid grid-cols-[minmax(0,8.5rem)_1fr_2.5rem] items-center gap-2">
            <span class="truncate" [title]="row.reply">{{ row.reply }}</span>
            <span class="h-3 rounded bg-muted">
              <span
                class="block h-3 rounded transition-[width] motion-reduce:transition-none"
                [class]="row.bad ? 'bg-fig-negative' : 'bg-fig-1'"
                [style.width.%]="row.p * 100"
              ></span>
            </span>
            <span class="text-right tabular-nums">{{ (row.p * 100).toFixed(0) }}%</span>
          </li>
        }
      </ul>

      <svg [attr.viewBox]="'0 0 ' + W + ' ' + H" class="mt-3 w-full" aria-hidden="true">
        <line
          [attr.x1]="32"
          [attr.x2]="W - 8"
          [attr.y1]="qy(0)"
          [attr.y2]="qy(0)"
          class="stroke-border"
        />
        <path [attr.d]="rewardLine" class="fill-none stroke-fig-4 stroke-2" />
        <path [attr.d]="qualityLine" class="fill-none stroke-fig-3 stroke-2" />
        <line
          [attr.x1]="cursor()"
          [attr.x2]="cursor()"
          y1="8"
          [attr.y2]="H - 22"
          class="stroke-muted-foreground"
          stroke-dasharray="2 2"
        />
        <text
          [attr.x]="(32 + W - 8) / 2"
          [attr.y]="H - 6"
          text-anchor="middle"
          class="fill-muted-foreground text-[8px]"
        >
          tight leash → loose leash
        </text>
      </svg>
      <p class="mt-1 flex flex-wrap gap-3 text-xs">
        <span class="text-fig-4">● reward score: {{ reward().toFixed(1) }}</span>
        <span class="text-fig-3">● real quality: {{ quality().toFixed(2) }}</span>
        <span class="text-muted-foreground"
          >base model's quality: {{ baseQuality.toFixed(2) }}</span
        >
      </p>
      <p class="mt-2 rounded-md bg-muted/50 p-2 text-sm" aria-live="polite">{{ verdict() }}</p>
      <p caption>
        Early on, the model drops wrong answers and the score and the quality rise together. Then it
        finds that more words earn more reward. Raters did like a few extra words, but no one ever
        rated the red reply, so the reward model never learnt it's awful. Only the green line tells
        you this is happening, and in real life nobody has the green line. That's why the leash
        exists, and why teams keep checking models against fresh human judgement.
      </p>
    </app-figure-frame>
  `,
})
export class RlhfLeashFigure {
  protected readonly W = W;
  protected readonly H = H;
  protected readonly qy = qy;
  protected readonly qualityLine = qualityLine;
  protected readonly rewardLine = rewardLine;
  protected readonly baseQuality = expected(BASE, TRUE_QUALITY);
  protected readonly looseness = signal(50);

  private readonly logBeta = computed(
    () => Math.log10(50) + (this.looseness() / 100) * (Math.log10(0.2) - Math.log10(50)),
  );
  protected readonly beta = computed(() => 10 ** this.logBeta());
  protected readonly cursor = computed(() => x(this.logBeta()));
  private readonly policy = computed(() => policyAt(this.logBeta()));
  protected readonly quality = computed(() => expected(this.policy(), TRUE_QUALITY));
  protected readonly reward = computed(() => expected(this.policy(), SCORES));
  protected readonly rows = computed(() =>
    REPLIES.map((reply, i) => ({ reply, p: this.policy()[i], bad: i === REPLIES.length - 1 })),
  );
  protected readonly summary = computed(
    () =>
      `Reward ${this.reward().toFixed(1)}, real quality ${this.quality().toFixed(2)}. ` +
      this.rows()
        .map((r) => `${r.reply}: ${(r.p * 100).toFixed(0)}%`)
        .join(', '),
  );
  protected readonly verdict = computed(() => {
    if (this.quality() > this.baseQuality + 0.2)
      return 'Better than the base model. RLHF is helping.';
    if (this.quality() >= this.baseQuality)
      return 'About the same as the base model. The leash is too tight to change much.';
    return "Worse than the base model, even though the reward score is higher. It's chasing the proxy.";
  });
}
