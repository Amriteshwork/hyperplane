import {
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  computed,
  inject,
  signal,
} from '@angular/core';

import { FigureFrame } from './figure-frame';
import { linear } from './lib/plot';
import { seeded } from './lib/random';
import {
  GROUPS,
  Group,
  SkipGram,
  createSkipGram,
  groupSeparation,
  skipGramStep,
} from './lib/word2vec';

const SIZE = 300;
const PER_FRAME = 150;
const END = 60_000;
const COLOR: Readonly<Record<Group, string>> = {
  animal: 'fill-fig-1',
  food: 'fill-fig-2',
  vehicle: 'fill-fig-3',
};
const GROUP_OF = new Map(
  (Object.entries(GROUPS) as [Group, readonly string[]][]).flatMap(([group, words]) =>
    words.map((w): [string, Group] => [w, group]),
  ),
);

/** Two-dimensional skip-gram training live in the page, so a reader can watch the groups form. */
@Component({
  selector: 'app-skipgram-trainer-figure',
  imports: [FigureFrame],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <app-figure-frame
      heading="Watch word2vec learn"
      hint="Press Train. Every word starts at a random spot."
    >
      <svg
        [attr.viewBox]="'0 0 ' + size + ' ' + size"
        class="mx-auto w-full max-w-md rounded-md bg-muted/40"
        role="img"
        [attr.aria-label]="
          'Word map after ' +
          steps() +
          ' training pairs. Group separation ' +
          separation().toFixed(2)
        "
      >
        @for (point of points(); track point.word) {
          @if (point.group) {
            <circle [attr.cx]="point.x" [attr.cy]="point.y" r="3.5" [class]="color(point.group)" />
            <text [attr.x]="point.x + 5" [attr.y]="point.y + 3" class="fill-foreground text-[9px]">
              {{ point.word }}
            </text>
          } @else {
            <circle
              [attr.cx]="point.x"
              [attr.cy]="point.y"
              r="1.5"
              class="fill-muted-foreground/50"
            />
          }
        }
      </svg>
      <p class="mt-2 flex flex-wrap justify-center gap-3 text-xs">
        <span class="text-fig-1">● animals</span>
        <span class="text-fig-2">● food</span>
        <span class="text-fig-3">● vehicles</span>
        <span class="text-muted-foreground">· other words</span>
      </p>

      <div class="mt-2 flex flex-wrap items-center gap-2">
        <button
          type="button"
          class="rounded-md border px-3 py-1.5 text-xs hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring"
          (click)="toggle()"
        >
          {{ running() ? 'Pause' : steps() ? 'Resume' : 'Train' }}
        </button>
        <button
          type="button"
          class="rounded-md border px-3 py-1.5 text-xs hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring"
          (click)="reset()"
        >
          Start over
        </button>
        <span class="text-xs tabular-nums text-muted-foreground" aria-live="off">
          {{ steps().toLocaleString('en') }} pairs · separation {{ separation().toFixed(2) }}
        </span>
      </div>
      <p caption>
        The model only ever sees pairs of nearby words, such as "cat" and "sleeps". It never sees
        the colours, or any word for "animal". Even so, the groups pull apart as it trains, because
        words in a group share neighbours. Separation is the average cosine similarity within a
        group minus the average between groups, and it starts near zero.
      </p>
    </app-figure-frame>
  `,
})
export class SkipgramTrainerFigure {
  protected readonly size = SIZE;
  protected readonly running = signal(false);
  protected readonly steps = signal(0);
  private readonly frame = signal(0);
  private rng = seeded(3);
  private model: SkipGram = createSkipGram(this.rng);
  private handle: number | null = null;

  protected readonly points = computed(() => {
    this.frame();
    const all = this.model.E.flat();
    const reach = Math.max(1, ...all.map(Math.abs));
    const scale = linear(-reach * 1.1, reach * 1.1, 12, SIZE - 30);
    return this.model.vocab.map((word, i) => ({
      word,
      group: GROUP_OF.get(word) ?? null,
      x: scale(this.model.E[i][0]),
      y: scale(this.model.E[i][1]),
    }));
  });
  protected readonly separation = computed(() => {
    this.frame();
    return groupSeparation(this.model);
  });

  constructor() {
    inject(DestroyRef).onDestroy(() => this.stop());
  }

  protected color(group: Group): string {
    return COLOR[group];
  }

  protected toggle(): void {
    if (this.running()) this.stop();
    else {
      if (this.steps() >= END) this.reset();
      this.running.set(true);
      this.handle = requestAnimationFrame(() => this.tick());
    }
  }

  protected reset(): void {
    this.stop();
    this.rng = seeded(3);
    this.model = createSkipGram(this.rng);
    this.steps.set(0);
    this.frame.update((f) => f + 1);
  }

  private tick(): void {
    for (let i = 0; i < PER_FRAME; i++) skipGramStep(this.model, this.rng);
    this.steps.update((s) => s + PER_FRAME);
    this.frame.update((f) => f + 1);
    if (this.steps() >= END) this.stop();
    else this.handle = requestAnimationFrame(() => this.tick());
  }

  private stop(): void {
    this.running.set(false);
    if (this.handle !== null) cancelAnimationFrame(this.handle);
    this.handle = null;
  }
}
