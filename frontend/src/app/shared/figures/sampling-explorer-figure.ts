import { ChangeDetectionStrategy, Component, computed, signal } from '@angular/core';

import { FigureFrame } from './figure-frame';
import { FigureSlider } from './figure-slider';
import { seeded } from './lib/random';
import { Knobs, drawToken, nextTokenProbs } from './lib/sampling';

/** Made-up but realistic odds for the next word after "the cat sat on the". */
const CHOICES = [
  { word: 'mat', p: 0.42 },
  { word: 'floor', p: 0.2 },
  { word: 'sofa', p: 0.14 },
  { word: 'roof', p: 0.09 },
  { word: 'bed', p: 0.07 },
  { word: 'moon', p: 0.04 },
  { word: 'keyboard', p: 0.03 },
  { word: 'purple', p: 0.01 },
] as const;
const LOGITS = CHOICES.map((c) => Math.log(c.p));

/** Three knobs on one distribution, with a button that draws from the result. */
@Component({
  selector: 'app-sampling-explorer-figure',
  imports: [FigureFrame, FigureSlider],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <app-figure-frame
      heading="“The cat sat on the …”: one distribution, three knobs"
      hint="Move the sliders, then draw 20 words."
    >
      <div class="grid gap-3 sm:grid-cols-3">
        <app-figure-slider
          label="Temperature τ"
          [(value)]="temperature"
          [min]="0.05"
          [max]="3"
          [step]="0.05"
          [readout]="temperature().toFixed(2)"
        />
        <app-figure-slider
          label="Top-k"
          [(value)]="topK"
          [min]="1"
          [max]="8"
          [readout]="topK() === 8 ? 'off' : '' + topK()"
        />
        <app-figure-slider
          label="Top-p"
          [(value)]="topP"
          [min]="0.1"
          [max]="1"
          [step]="0.05"
          [readout]="topP() === 1 ? 'off' : topP().toFixed(2)"
        />
      </div>

      <ul class="list-none pl-0 mt-3 space-y-1 text-xs" role="img" [attr.aria-label]="summary()">
        @for (row of rows(); track row.word) {
          <li class="grid grid-cols-[4.5rem_1fr_3rem] items-center gap-2">
            <span class="truncate font-mono" [class.opacity-40]="row.after === 0">{{
              row.word
            }}</span>
            <span class="relative h-4 rounded bg-muted">
              <span
                class="absolute inset-y-0 left-0 rounded border-r border-muted-foreground/60"
                [style.width.%]="row.before * 100"
              ></span>
              <span
                class="absolute inset-y-0 left-0 rounded bg-fig-1 transition-[width] motion-reduce:transition-none"
                [style.width.%]="row.after * 100"
              ></span>
            </span>
            <span class="text-right tabular-nums">{{ (row.after * 100).toFixed(0) }}%</span>
          </li>
        }
      </ul>
      <p class="mt-1 text-xs text-muted-foreground">
        The thin mark is the model's own odds. {{ kept() }} of 8 words can still be drawn.
      </p>

      <div class="mt-3 flex flex-wrap items-center gap-2">
        <button
          type="button"
          class="rounded-md border px-3 py-1.5 text-xs hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring"
          (click)="draw()"
        >
          Draw 20 words
        </button>
        <button
          type="button"
          class="rounded-md border px-3 py-1.5 text-xs hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring"
          (click)="reset()"
        >
          Reset
        </button>
      </div>
      <p class="mt-2 min-h-5 font-mono text-xs leading-5" aria-live="polite">
        {{ drawn().join(' ') }}
      </p>
      <p caption>
        Slide τ near 0.05 and draw: "mat" every time. Slide it to 3 and "purple" starts to appear.
        Temperature changes every word's odds, while top-k and top-p only remove the unlikely end.
        Now set τ to 3 and top-p to 0.5: the odds are flat, so top-p keeps many words. Set τ to 0.5
        with top-p still at 0.5: the leader is strong, so it keeps one or two. That's the difference
        from top-k, which always keeps the same number.
      </p>
    </app-figure-frame>
  `,
})
export class SamplingExplorerFigure {
  protected readonly temperature = signal(1);
  protected readonly topK = signal(8);
  protected readonly topP = signal(1);
  protected readonly drawn = signal<readonly string[]>([]);
  private readonly rng = seeded(11);

  private readonly knobs = computed((): Knobs => ({
    temperature: this.temperature(),
    topK: this.topK() === 8 ? null : this.topK(),
    topP: this.topP() === 1 ? null : this.topP(),
  }));
  private readonly probs = computed(() => nextTokenProbs(LOGITS, this.knobs()));
  protected readonly rows = computed(() =>
    CHOICES.map((c, i) => ({ word: c.word, before: c.p, after: this.probs()[i] })),
  );
  protected readonly kept = computed(() => this.probs().filter((p) => p > 0).length);
  protected readonly summary = computed(
    () =>
      'Next word odds: ' +
      this.rows()
        .map((r) => `${r.word} ${(r.after * 100).toFixed(0)}%`)
        .join(', '),
  );

  protected draw(): void {
    const p = this.probs();
    this.drawn.set(Array.from({ length: 20 }, () => CHOICES[drawToken(p, this.rng())].word));
  }

  protected reset(): void {
    this.temperature.set(1);
    this.topK.set(8);
    this.topP.set(1);
    this.drawn.set([]);
  }
}
