import { ChangeDetectionStrategy, Component, computed, signal } from '@angular/core';

import { FigureFrame } from './figure-frame';
import { CROSS_ATTENTION } from './lib/measured';

type Stage = 'encode' | 'self' | 'cross' | 'write';

const STAGE_TEXT: Readonly<Record<Stage, string>> = {
  encode:
    'The encoder reads the whole English sentence with no mask, once. Its vectors are the memory.',
  self: 'Masked self-attention: this position looks back at the German written so far, and never ahead.',
  cross: 'Cross-attention: this position asks the English which words matter now.',
  write: 'The MLP and read-out turn that into next-word scores. The likeliest word is appended.',
};

/** Step through a translation: one encoder pass, then three kinds of attention for each word. */
@Component({
  selector: 'app-seq2seq-steps-figure',
  imports: [FigureFrame],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <app-figure-frame
      heading="One translation, step by step"
      hint="Step through and watch the three kinds of attention take turns."
    >
      <p class="text-xs font-semibold">Encoder</p>
      <div class="mt-1 flex flex-wrap gap-1">
        @for (word of data.english; track $index; let s = $index) {
          <span
            class="rounded-md border px-1.5 py-1 font-mono text-xs"
            [class.bg-fig-1/15]="stage() === 'encode'"
            [style.background-color]="readColor(s)"
            >{{ word }}</span
          >
        }
      </div>

      <p class="mt-3 text-xs font-semibold">
        Decoder: {{ written() }} of {{ data.german.length }} words
      </p>
      <div class="mt-1 flex min-h-8 flex-wrap gap-1">
        <span class="rounded-md border px-1.5 py-1 font-mono text-xs text-muted-foreground"
          >[BOS]</span
        >
        @for (word of data.german; track $index; let t = $index) {
          @if (t < written()) {
            <span
              class="rounded-md border px-1.5 py-1 font-mono text-xs"
              [class.bg-fig-2/20]="stage() === 'self'"
              >{{ word }}</span
            >
          }
        }
      </div>

      <p class="mt-3 rounded-md bg-muted/50 p-2 text-sm" aria-live="polite">
        <span class="font-semibold">Step {{ at() + 1 }} of {{ total }}.</span> {{ stageText() }}
      </p>
      <div class="mt-2 flex flex-wrap gap-1.5">
        <button
          type="button"
          class="rounded-md border px-3 py-1.5 text-xs hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring disabled:opacity-50"
          [disabled]="at() === total - 1"
          (click)="at.set(at() + 1)"
        >
          Next step
        </button>
        <button
          type="button"
          class="rounded-md border px-3 py-1.5 text-xs hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring"
          (click)="at.set(0)"
        >
          Restart
        </button>
      </div>
      <p caption>
        The encoder runs once, however long the translation. The decoder runs once per word, and it
        rereads the same memory each time, with a different question. In practice the decoder's
        earlier work is cached, as on the Text generation page.
      </p>
    </app-figure-frame>
  `,
})
export class Seq2seqStepsFigure {
  protected readonly data = CROSS_ATTENTION;
  /** One encoder step, then three steps per German word. */
  protected readonly total = 1 + 3 * this.data.german.length;
  protected readonly at = signal(0);
  protected readonly stage = computed((): Stage => {
    if (this.at() === 0) return 'encode';
    return (['self', 'cross', 'write'] as const)[(this.at() - 1) % 3];
  });
  private readonly word = computed(() => Math.floor((this.at() - 1) / 3));
  protected readonly written = computed(() =>
    this.at() === 0 ? 0 : this.word() + (this.stage() === 'write' ? 1 : 0),
  );
  protected readonly stageText = computed(() => {
    const base = STAGE_TEXT[this.stage()];
    return this.stage() === 'encode' ? base : `Writing "${this.data.german[this.word()]}". ${base}`;
  });

  /** During cross-attention, shade the English by how much the current word reads it. */
  protected readColor(s: number): string | null {
    if (this.stage() !== 'cross') return null;
    const weight = this.data.weights[this.word()][s];
    return `color-mix(in srgb, var(--fig-1) ${Math.round(weight * 100)}%, transparent)`;
  }
}
