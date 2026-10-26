import { ChangeDetectionStrategy, Component, computed, signal } from '@angular/core';

import { FigureFrame } from './figure-frame';
import { FigureSlider } from './figure-slider';
import { linear, path } from './lib/plot';
import { encoding, frequency, similarity } from './lib/positional';

const D = 32;
const POSITIONS = 48;
const CLOCKS = [0, 1, 3, 8]; // which (sin, cos) pairs get a clock face
const TABLE = Array.from({ length: POSITIONS }, (_, t) => encoding(t, D));
// Heatmap: one row per position, one column per dimension.
const MAP = { left: 26, top: 6, cell: 10, row: 4 };
// Similarity chart.
const sx = linear(0, POSITIONS - 1, 30, 350);
const sy = linear(-6, D / 2, 120, 10);

/** Sinusoidal encodings as clocks, as a table, and as a similarity that only cares about distance. */
@Component({
  selector: 'app-positional-encoding-explorer-figure',
  imports: [FigureFrame, FigureSlider],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <app-figure-frame
      heading="Positions as a row of clocks"
      hint="Move the position: fast clocks tell neighbours apart, slow clocks tell far-apart positions apart."
    >
      <app-figure-slider label="Position t" [(value)]="t" [min]="0" [max]="last" />

      <svg viewBox="0 0 360 100" class="mt-3 w-full" role="img" [attr.aria-label]="clockText()">
        @for (clock of clocks(); track clock.pair; let i = $index) {
          <g [attr.transform]="'translate(' + (45 + i * 90) + ', 40)'">
            <circle r="26" class="fill-card stroke-border" stroke-width="1.5" />
            <line
              x1="0"
              y1="0"
              [attr.x2]="22 * clock.sin"
              [attr.y2]="-22 * clock.cos"
              class="stroke-fig-1"
              stroke-width="3"
              stroke-linecap="round"
            />
            <circle r="2.5" class="fill-fig-1" />
            <text y="40" text-anchor="middle" class="fill-foreground text-[8px]">
              pair {{ clock.pair }}
            </text>
            <text y="50" text-anchor="middle" class="fill-muted-foreground text-[8px]">
              1 turn / {{ clock.period }} tokens
            </text>
          </g>
        }
      </svg>

      <svg
        [attr.viewBox]="'0 0 360 ' + mapHeight"
        class="mt-2 w-full"
        role="img"
        aria-label="Heatmap of the encoding: one row per position, one column per dimension. Left columns change fast down the rows, right columns barely change."
      >
        @for (row of table; track $index; let r = $index) {
          @for (v of row; track $index; let c = $index) {
            <rect
              [attr.x]="map.left + c * map.cell"
              [attr.y]="map.top + r * map.row"
              [attr.width]="map.cell"
              [attr.height]="map.row"
              [class]="v >= 0 ? 'fill-fig-1' : 'fill-fig-2'"
              [attr.fill-opacity]="Math.abs(v)"
            />
          }
        }
        <rect
          x="0"
          [attr.y]="map.top + t() * map.row - 1"
          [attr.width]="map.left + D * map.cell + 2"
          [attr.height]="map.row + 2"
          class="fill-none stroke-foreground"
          stroke-width="1.2"
        />
        <text x="0" [attr.y]="map.top + t() * map.row + 4" class="fill-foreground text-[8px]">
          t={{ t() }}
        </text>
        <text [attr.x]="map.left" [attr.y]="mapHeight - 2" class="fill-muted-foreground text-[8px]">
          dimension 0 (fast) →
        </text>
        <text
          [attr.x]="map.left + D * map.cell"
          [attr.y]="mapHeight - 2"
          text-anchor="end"
          class="fill-muted-foreground text-[8px]"
        >
          → dimension 31 (slow)
        </text>
      </svg>

      <p class="mt-3 text-xs">
        <span class="font-semibold text-fig-1">Similarity p_t · p_s</span>
        <span class="text-muted-foreground"> of position t to every position s</span>
      </p>
      <svg viewBox="0 0 360 140" class="w-full" role="img" [attr.aria-label]="curveText()">
        <line x1="30" x2="350" [attr.y1]="sy(0)" [attr.y2]="sy(0)" class="stroke-border" />
        <line
          [attr.x1]="sx(t())"
          [attr.x2]="sx(t())"
          y1="10"
          y2="120"
          class="stroke-fig-2"
          stroke-dasharray="3 3"
        />
        <path [attr.d]="curve()" class="fill-none stroke-fig-1" stroke-width="2" />
        <text x="30" y="134" class="fill-muted-foreground text-[8px]">position s = 0</text>
        <text x="350" y="134" text-anchor="end" class="fill-muted-foreground text-[8px]">
          s = {{ last }}
        </text>
      </svg>

      <p caption>
        Each clock is one (sin, cos) pair of the 32-number encoding, turning at its own speed: the
        first once every 6 tokens, the slowest one shown about once every 600. The heatmap shows the
        same thing for positions 0–47: blue for positive, orange for negative, stripes fastest on
        the left. In the bottom chart, as you move t, the curve's shape doesn't change, it just
        slides along: how similar two positions look depends only on how far apart they are, never
        on where they are. That is why "the word 3 back" is equally easy to find anywhere in a
        sentence.
      </p>
    </app-figure-frame>
  `,
})
export class PositionalEncodingExplorerFigure {
  protected readonly Math = Math;
  protected readonly D = D;
  protected readonly last = POSITIONS - 1;
  protected readonly table = TABLE;
  protected readonly map = MAP;
  protected readonly mapHeight = MAP.top + POSITIONS * MAP.row + 14;
  protected readonly sx = sx;
  protected readonly sy = sy;
  protected readonly t = signal(10);

  protected readonly clocks = computed(() =>
    CLOCKS.map((pair) => {
      const angle = this.t() * frequency(pair, D);
      return {
        pair,
        sin: Math.sin(angle),
        cos: Math.cos(angle),
        period: Math.round((2 * Math.PI) / frequency(pair, D)),
      };
    }),
  );

  protected readonly curve = computed(() =>
    path(TABLE.map((_, s) => [sx(s), sy(similarity(this.t(), s, D))] as const)),
  );

  protected readonly clockText = computed(
    () =>
      `Clocks at position ${this.t()}: ` +
      this.clocks()
        .map((c) => `pair ${c.pair} (one turn per ${c.period} tokens)`)
        .join(', ') +
      '.',
  );

  protected readonly curveText = computed(() => {
    const t = this.t();
    const other = t + 3 <= this.last ? t + 3 : t - 3; // a position 3 away
    return (
      `Similarity of position ${t} to every position: highest (${D / 2}) at itself, falling off ` +
      `with distance on both sides; ${t} vs ${other}: ${similarity(t, other, D).toFixed(2)}.`
    );
  });
}
