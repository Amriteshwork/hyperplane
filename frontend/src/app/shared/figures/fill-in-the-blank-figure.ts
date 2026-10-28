import { ChangeDetectionStrategy, Component, computed, signal } from '@angular/core';

import { FigureFrame } from './figure-frame';
import { FOODS, NAMES, corpusSentence, recover } from './lib/bert';
import { seeded } from './lib/random';

/** Hide words of a generated sentence and see which reader, left-to-right or BERT, can fill them. */
@Component({
  selector: 'app-fill-in-the-blank-figure',
  imports: [FigureFrame],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <app-figure-frame
      heading="Fill in the blanks: one way or both ways"
      hint="Click words to hide them. Then switch the reader."
    >
      <div class="flex flex-wrap gap-1.5" role="group" aria-label="Words of the sentence">
        @for (word of words(); track $index; let i = $index) {
          <button
            type="button"
            class="min-w-10 rounded-md border px-2 py-1.5 font-mono text-sm transition-colors focus-visible:outline-2 focus-visible:outline-ring motion-reduce:transition-none"
            [class]="wordClass(i)"
            [attr.aria-pressed]="hidden().has(i)"
            [attr.aria-label]="(hidden().has(i) ? 'Hidden: ' : '') + word.text"
            (click)="toggle(i)"
          >
            {{ hidden().has(i) ? '[MASK]' : word.text }}
          </button>
        }
      </div>
      <p class="mt-2 text-xs text-muted-foreground">
        Outlined: what the reader can use for the word you last hid. Words 5 and 7 repeat words 0
        and 2.
      </p>

      <div class="mt-3 flex flex-wrap gap-2" role="group" aria-label="Reader">
        @for (option of readers; track option.bidirectional) {
          <button
            type="button"
            class="rounded-md border px-3 py-1.5 text-xs hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring"
            [class.bg-muted]="bidirectional() === option.bidirectional"
            [attr.aria-pressed]="bidirectional() === option.bidirectional"
            (click)="bidirectional.set(option.bidirectional)"
          >
            {{ option.label }}
          </button>
        }
        <button
          type="button"
          class="rounded-md border px-3 py-1.5 text-xs hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring"
          (click)="newSentence()"
        >
          New sentence
        </button>
      </div>

      <div class="mt-3 rounded-md bg-muted/50 p-3 text-sm" aria-live="polite">
        @if (focus(); as f) {
          <p>
            <span class="font-mono font-semibold">[MASK]</span> at position {{ f.at }}:
            {{ f.recovery.reason }}
          </p>
        } @else {
          <p>Hide a word to see how each reader fills it in.</p>
        }
        @if (hidden().size) {
          <p class="mt-1 text-xs text-muted-foreground">
            Expected words right: <span class="text-foreground">{{ score() }}</span> of
            {{ hidden().size }} hidden. The other reader would get {{ otherScore() }}.
          </p>
        }
      </div>
      <p caption>
        Try hiding the first name. Reading left to right, the reader has nothing yet, so it can only
        guess one name in eight. Reading both ways, it finds the copy at position 5 and gets it
        right. Hide position 5 instead and both readers succeed, because the copy is now on the
        left. Hide both, and nobody can know. That is why BERT is trained by masking: the answer has
        to be taken out of the input, because a reader that sees both sides would otherwise just
        look at it.
      </p>
    </app-figure-frame>
  `,
})
export class FillInTheBlankFigure {
  protected readonly readers = [
    { label: 'Left to right (GPT)', bidirectional: false },
    { label: 'Both ways (BERT)', bidirectional: true },
  ];

  private readonly rng = seeded(7);
  private readonly pair = signal<[string, string]>(['zoe', 'jam']);
  protected readonly words = computed(() => corpusSentence(...this.pair()));
  protected readonly hidden = signal<ReadonlySet<number>>(new Set([0]));
  private readonly last = signal<number | null>(0);
  protected readonly bidirectional = signal(true);

  protected readonly focus = computed(() => {
    const at = this.last();
    if (at === null || !this.hidden().has(at)) return null;
    return { at, recovery: recover(this.words(), this.hidden(), at, this.bidirectional()) };
  });

  private expected(bidirectional: boolean): string {
    let total = 0;
    for (const at of this.hidden())
      total += recover(this.words(), this.hidden(), at, bidirectional).probability;
    return Number.isInteger(total) ? String(total) : total.toFixed(2);
  }

  protected readonly score = computed(() => this.expected(this.bidirectional()));
  protected readonly otherScore = computed(() => this.expected(!this.bidirectional()));

  protected toggle(i: number): void {
    const next = new Set(this.hidden());
    if (next.has(i)) {
      next.delete(i);
      this.last.set([...next].at(-1) ?? null);
    } else {
      next.add(i);
      this.last.set(i);
    }
    this.hidden.set(next);
  }

  protected newSentence(): void {
    const pick = <T>(xs: readonly T[]): T => xs[Math.floor(this.rng() * xs.length)];
    this.pair.set([pick(NAMES), pick(FOODS)]);
  }

  protected wordClass(i: number): string {
    const f = this.focus();
    if (this.hidden().has(i))
      return f?.at === i ? 'border-fig-2 bg-fig-2/15 font-semibold' : 'border-fig-2/60 bg-muted';
    if (f?.recovery.source === i) return 'border-fig-3 bg-fig-3/20 ring-2 ring-fig-3';
    if (f && f.recovery.visible[i]) return 'border-fig-1 bg-fig-1/10';
    if (f) return 'border-dashed opacity-40';
    return 'hover:bg-muted';
  }
}
