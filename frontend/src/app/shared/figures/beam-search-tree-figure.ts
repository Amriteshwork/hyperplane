import { ChangeDetectionStrategy, Component, computed, signal } from '@angular/core';

import { FigureFrame } from './figure-frame';
import { TREE, beamSteps } from './lib/sampling';

const W = 320;
const H = 200;

interface TreeNode {
  key: string;
  word: string;
  x: number;
  y: number;
  parent: string | null;
  p: number;
  depth: number;
}

function layout(): TreeNode[] {
  const nodes: TreeNode[] = [
    { key: 'start', word: 'start', x: 30, y: H / 2, parent: null, p: 1, depth: 0 },
  ];
  const firsts = Object.entries(TREE['start']);
  let row = 0;
  const rows = firsts.reduce((sum, [word]) => sum + Object.keys(TREE[word]).length, 0);
  firsts.forEach(([first, p]) => {
    const children = Object.entries(TREE[first]);
    const ys = children.map(() => 12 + ((row++ + 0.5) / rows) * (H - 24));
    nodes.push({
      key: first,
      word: first,
      x: 140,
      y: ys.reduce((a, b) => a + b, 0) / ys.length,
      parent: 'start',
      p,
      depth: 1,
    });
    children.forEach(([second, q], i) =>
      nodes.push({
        key: `${first} ${second}`,
        word: second,
        x: 250,
        y: ys[i],
        parent: first,
        p: q,
        depth: 2,
      }),
    );
  });
  return nodes;
}

const NODES = layout();
const BY_KEY = new Map(NODES.map((n) => [n.key, n]));

/** Step a beam through a small tree and see a wider beam find a sentence greedy misses. */
@Component({
  selector: 'app-beam-search-tree-figure',
  imports: [FigureFrame],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <app-figure-frame
      heading="Beam search, step by step"
      hint="Choose a width, then step through the search."
    >
      <div class="flex flex-wrap items-center gap-1.5" role="group" aria-label="Beam width">
        @for (option of [1, 2, 3]; track option) {
          <button
            type="button"
            class="rounded-md border px-3 py-1.5 text-xs hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring"
            [class.bg-muted]="width() === option"
            [attr.aria-pressed]="width() === option"
            (click)="setWidth(option)"
          >
            Width {{ option }}{{ option === 1 ? ' (greedy)' : '' }}
          </button>
        }
        <button
          type="button"
          class="rounded-md border px-3 py-1.5 text-xs hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring disabled:opacity-50"
          [disabled]="step() === 2"
          (click)="step.set(step() + 1)"
        >
          Next step
        </button>
        <button
          type="button"
          class="rounded-md border px-3 py-1.5 text-xs hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring"
          (click)="step.set(0)"
        >
          Restart
        </button>
      </div>

      <svg
        [attr.viewBox]="'0 0 ' + W + ' ' + H"
        class="mt-2 w-full"
        role="img"
        [attr.aria-label]="status()"
      >
        @for (node of nodes; track node.key) {
          @if (node.parent; as parent) {
            @if (edgeOf(node); as e) {
              <line
                [attr.x1]="e.x + 22"
                [attr.y1]="e.y"
                [attr.x2]="node.x - 22"
                [attr.y2]="node.y"
                [class]="isKept(node.key) ? 'stroke-fig-1 stroke-2' : 'stroke-border'"
              />
              <text
                [attr.x]="(e.x + node.x) / 2"
                [attr.y]="(e.y + node.y) / 2 - 3"
                text-anchor="middle"
                class="fill-muted-foreground text-[8px]"
              >
                {{ node.p }}
              </text>
            }
          }
        }
        @for (node of nodes; track node.key) {
          <g [class.opacity-30]="node.depth > step()">
            <rect
              [attr.x]="node.x - 22"
              [attr.y]="node.y - 9"
              width="44"
              height="18"
              rx="4"
              [class]="nodeClass(node)"
            />
            <text
              [attr.x]="node.x"
              [attr.y]="node.y + 3"
              text-anchor="middle"
              class="fill-foreground text-[9px]"
            >
              {{ node.word }}
            </text>
            @if (node.depth === 2 && step() === 2 && isKept(node.key)) {
              <text [attr.x]="node.x + 26" [attr.y]="node.y + 3" class="fill-foreground text-[8px]">
                {{ score(node.key) }}
              </text>
            }
          </g>
        }
      </svg>

      <p class="mt-2 rounded-md bg-muted/50 p-2 text-sm" aria-live="polite">{{ status() }}</p>
      <p caption>
        At width 1 the search takes "the" because 0.5 beats 0.4, and it can never go back. It ends
        on "the cat" at 0.20. At width 2 it keeps "a" alive too, and "a cat" scores 0.4 × 0.9 =
        0.36. Going to width 3 finds nothing better here, so the extra cost buys nothing. That's the
        trade-off a real system tunes.
      </p>
    </app-figure-frame>
  `,
})
export class BeamSearchTreeFigure {
  protected readonly W = W;
  protected readonly H = H;
  protected readonly nodes = NODES;
  protected readonly width = signal(1);
  protected readonly step = signal(0);

  private readonly beams = computed(() => beamSteps(this.width()));
  private readonly kept = computed(() => {
    const keys = new Set<string>(['start']);
    for (let s = 1; s <= this.step(); s++)
      for (const beam of this.beams()[s])
        beam.words.forEach((_, i) => keys.add(beam.words.slice(0, i + 1).join(' ')));
    return keys;
  });

  protected readonly status = computed(() => {
    const s = this.step();
    if (s === 0) return 'Start: nothing has been written yet.';
    const beams = this.beams()[s];
    const kept = beams.map((b) => `"${b.words.join(' ')}" ${b.probability.toFixed(2)}`).join(', ');
    if (s < 2) return `After one word, keep ${beams.length}: ${kept}.`;
    return `Finished. Kept: ${kept}. Best: "${beams[0].words.join(' ')}" at ${beams[0].probability.toFixed(2)}.`;
  });

  protected setWidth(width: number): void {
    this.width.set(width);
    this.step.set(0);
  }

  protected edgeOf(node: TreeNode): TreeNode | undefined {
    return node.parent ? BY_KEY.get(node.parent) : undefined;
  }

  protected isKept(key: string): boolean {
    return this.kept().has(key);
  }

  protected score(key: string): string {
    const beam = this.beams()[2].find((b) => b.words.join(' ') === key);
    return beam ? beam.probability.toFixed(2) : '';
  }

  protected nodeClass(node: TreeNode): string {
    return this.isKept(node.key) ? 'fill-fig-1/25 stroke-fig-1' : 'fill-muted stroke-border';
  }
}
