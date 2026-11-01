import { ChangeDetectionStrategy, Component, computed, signal } from '@angular/core';

import { FigureFrame } from './figure-frame';
import { FigureSlider } from './figure-slider';
import { fromLogScores, logNormal } from './lib/naive-bayes';

const CENTRES = [-0.5, 0.5];
const WIDTH = 460;

/** The same feature repeated: no new information, but the naive model grows more certain. */
@Component({
  selector: 'app-double-counting-figure',
  imports: [FigureFrame, FigureSlider],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <app-figure-frame heading="Counting one clue many times" hint="Add copies of the same feature.">
      <svg
        [attr.viewBox]="'0 0 ' + width + ' 92'"
        class="w-full rounded-md border"
        role="img"
        [attr.aria-label]="description()"
      >
        <text x="4" y="14" class="fill-muted-foreground text-[10px]">True probability</text>
        <rect x="0" y="20" [attr.width]="width" height="20" class="fill-muted" />
        <rect x="0" y="20" [attr.width]="(width * truth()) / 100" height="20" class="fill-fig-3" />
        <text x="4" y="60" class="fill-muted-foreground text-[10px]">
          Naive model's probability
        </text>
        <rect x="0" y="66" [attr.width]="width" height="20" class="fill-muted" />
        <rect x="0" y="66" [attr.width]="(width * naive()) / 100" height="20" class="fill-fig-1" />
      </svg>
      <div class="mt-3 grid gap-3 sm:grid-cols-2">
        <app-figure-slider label="Copies of the feature" [(value)]="copies" [min]="1" [max]="20" />
        <app-figure-slider
          label="Its value"
          [(value)]="value"
          [min]="-2"
          [max]="2"
          [step]="0.05"
          [readout]="value().toFixed(2)"
        />
      </div>
      <dl class="mt-2 grid grid-cols-2 gap-2 text-xs" aria-live="polite">
        <div>
          <dt class="text-muted-foreground">P(class B), truly</dt>
          <dd class="font-mono tabular-nums">
            <span data-testid="truth">{{ truth().toFixed(1) }}</span
            >%
          </dd>
        </div>
        <div>
          <dt class="text-muted-foreground">P(class B), says the model</dt>
          <dd class="font-mono tabular-nums">
            <span data-testid="naive">{{ naive().toFixed(1) }}</span
            >%
          </dd>
        </div>
      </dl>
      <p caption>
        Two equally common classes have bell curves centred at −0.5 and +0.5. With one copy, the
        model is exactly right. Each extra copy multiplies in the same evidence, so the model's
        answer heads to 0 or 100% while the truth stays put. It doesn't change which class wins
        though: every copy favours the same one. Correlated features do the same thing more gently.
      </p>
    </app-figure-frame>
  `,
})
export class DoubleCountingFigure {
  protected readonly width = WIDTH;
  protected readonly copies = signal(1);
  protected readonly value = signal(0.5);

  private probability(copies: number): number {
    const scores = CENTRES.map((c) => copies * logNormal(this.value(), c, 1));
    return 100 * fromLogScores(scores)[1];
  }

  protected readonly truth = computed(() => this.probability(1));
  protected readonly naive = computed(() => this.probability(this.copies()));
  protected readonly description = computed(
    () =>
      `With ${this.copies()} copies of a feature at ${this.value().toFixed(2)}, the true ` +
      `probability of class B is ${this.truth().toFixed(1)}% and the naive model says ` +
      `${this.naive().toFixed(1)}%.`,
  );
}
