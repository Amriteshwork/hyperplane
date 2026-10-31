import { ChangeDetectionStrategy, Component, computed, signal } from '@angular/core';

import { FigureFrame } from './figure-frame';
import { linear } from './lib/plot';
import { ANALOGIES, WORD_LIST, nearestWords, pca2d, solveAnalogy } from './lib/word2vec';

const W = 340;
const H = 240;
const XY = pca2d(WORD_LIST);
const xs = WORD_LIST.map((w) => XY[w][0]);
const ys = WORD_LIST.map((w) => XY[w][1]);
const sx = linear(Math.min(...xs), Math.max(...xs), 14, W - 34);
const sy = linear(Math.min(...ys), Math.max(...ys), H - 12, 12);

/** Real GloVe vectors on a map: nearest neighbours and word arithmetic, on all 50 dimensions. */
@Component({
  selector: 'app-word-map-figure',
  imports: [FigureFrame],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <app-figure-frame
      heading="Real word vectors: neighbours and arithmetic"
      hint="Click a dot, choose a word from the list, or try an analogy."
    >
      <svg
        [attr.viewBox]="'0 0 ' + W + ' ' + H"
        class="w-full rounded-md bg-muted/40"
        role="img"
        [attr.aria-label]="summary()"
      >
        @if (arrows(); as a) {
          <line
            [attr.x1]="a.ax"
            [attr.y1]="a.ay"
            [attr.x2]="a.bx"
            [attr.y2]="a.by"
            class="stroke-fig-2 stroke-[1.5]"
          />
          <line
            [attr.x1]="a.cx"
            [attr.y1]="a.cy"
            [attr.x2]="a.dx"
            [attr.y2]="a.dy"
            class="stroke-fig-3 stroke-[1.5]"
          />
        }
        @for (point of points; track point.word) {
          <g class="cursor-pointer" (click)="pick(point.word)">
            <circle [attr.cx]="point.x" [attr.cy]="point.y" r="6" fill="transparent" />
            <circle
              [attr.cx]="point.x"
              [attr.cy]="point.y"
              [attr.r]="isLit(point.word) ? 3.5 : 1.8"
              [class]="isLit(point.word) ? 'fill-fig-1' : 'fill-muted-foreground/60'"
            />
            @if (isLit(point.word)) {
              <text
                [attr.x]="point.x + 4"
                [attr.y]="point.y + 3"
                class="fill-foreground text-[8px] font-semibold"
              >
                {{ point.word }}
              </text>
            }
          </g>
        }
      </svg>
      <p class="mt-1 text-xs text-muted-foreground">
        50 dimensions drawn in 2, so closeness on this map is only rough. The lists below use all
        50.
      </p>

      <div class="mt-3 grid gap-3 sm:grid-cols-2">
        <div>
          <label class="block text-xs font-semibold" for="word-map-select">Nearest to</label>
          <select
            id="word-map-select"
            class="mt-1 w-full rounded-md border bg-background p-1.5 text-sm"
            [value]="word()"
            (change)="onSelect($event)"
          >
            @for (w of words; track w) {
              <option [value]="w" [selected]="w === word()">{{ w }}</option>
            }
          </select>
          <ol class="list-none pl-0 mt-2 space-y-1 text-xs" aria-live="polite">
            @for (near of near(); track near.word) {
              <li class="grid grid-cols-[4.5rem_1fr_2.2rem] items-center gap-1.5">
                <span class="truncate font-mono">{{ near.word }}</span>
                <span class="h-2 rounded bg-muted">
                  <span class="block h-2 rounded bg-fig-1" [style.width.%]="near.cos * 100"></span>
                </span>
                <span class="text-right tabular-nums">{{ near.cos.toFixed(2) }}</span>
              </li>
            }
          </ol>
        </div>
        <div>
          <p class="text-xs font-semibold">Word arithmetic</p>
          <div class="mt-1 flex flex-wrap gap-1" role="group" aria-label="Analogies">
            @for (analogy of analogies; track $index; let i = $index) {
              <button
                type="button"
                class="rounded-md border px-1.5 py-1 text-[11px] hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring"
                [class.bg-muted]="at() === i"
                [attr.aria-pressed]="at() === i"
                (click)="at.set(i)"
              >
                {{ analogy[1] }} − {{ analogy[0] }} + {{ analogy[2] }}
              </button>
            }
          </div>
          <p class="mt-2 text-sm" aria-live="polite">
            <span class="font-mono text-xs"
              >{{ current()[1] }} − {{ current()[0] }} + {{ current()[2] }}</span
            >
            ≈ <span class="font-semibold">{{ answers()[0].word }}</span>
          </p>
          <p class="mt-1 text-xs text-muted-foreground">
            Runners-up:
            @for (a of answers().slice(1); track a.word; let last = $last) {
              {{ a.word }} {{ a.cos.toFixed(2) }}{{ last ? '' : ',' }}
            }
          </p>
        </div>
      </div>
      <p caption>
        These are real GloVe vectors, learnt from billions of words of text. Pick "monday" and the
        other days appear. Pick "piano" and you get instruments. Nobody labelled these. The orange
        and green lines show the arithmetic: on the map they're only roughly parallel, because two
        dimensions lose a lot, but in 50 dimensions the sum lands closest to the answer. This search
        covers only the words on the map and leaves out the three input words.
      </p>
    </app-figure-frame>
  `,
})
export class WordMapFigure {
  protected readonly W = W;
  protected readonly H = H;
  protected readonly words = WORD_LIST;
  protected readonly analogies = ANALOGIES;
  protected readonly points = WORD_LIST.map((word) => ({
    word,
    x: sx(XY[word][0]),
    y: sy(XY[word][1]),
  }));
  protected readonly word = signal('king');
  protected readonly at = signal(0);
  protected readonly near = computed(() => nearestWords(this.word()));
  protected readonly current = computed(() => this.analogies[this.at()]);
  protected readonly answers = computed(() => {
    const [a, b, c] = this.current();
    return solveAnalogy(a, b, c);
  });
  private readonly lit = computed(
    () =>
      new Set([
        this.word(),
        ...this.near().map((n) => n.word),
        ...this.current(),
        this.answers()[0].word,
      ]),
  );
  protected readonly arrows = computed(() => {
    const [a, b, c] = this.current();
    const d = this.answers()[0].word;
    const at = (w: string): [number, number] => [sx(XY[w][0]), sy(XY[w][1])];
    const [ax, ay] = at(a);
    const [bx, by] = at(b);
    const [cx, cy] = at(c);
    const [dx, dy] = at(d);
    return { ax, ay, bx, by, cx, cy, dx, dy };
  });
  protected readonly summary = computed(() => {
    const [a, b, c] = this.current();
    return `Nearest to ${this.word()}: ${this.near()
      .map((n) => n.word)
      .join(', ')}. ${b} minus ${a} plus ${c} is closest to ${this.answers()[0].word}.`;
  });

  protected isLit(word: string): boolean {
    return this.lit().has(word);
  }

  protected pick(word: string): void {
    this.word.set(word);
  }

  protected onSelect(event: Event): void {
    this.word.set((event.target as HTMLSelectElement).value);
  }
}
