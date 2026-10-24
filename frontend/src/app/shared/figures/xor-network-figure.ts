import { DecimalPipe } from '@angular/common';
import { ChangeDetectionStrategy, Component, computed, signal } from '@angular/core';

import { FigureFrame } from './figure-frame';
import { FigureSlider } from './figure-slider';
import { sigmoid } from './lib/tiny-mlp';

interface Neuron {
  id: string;
  label: string;
  x: number;
  y: number;
}

interface Edge {
  from: string;
  to: string;
  weight: number;
}

// The neurons page's hand-set network: hidden OR and AND, output "OR and not AND".
const HIDDEN = [
  { w: [20, 20], b: -10 },
  { w: [20, 20], b: -30 },
];
const OUTPUT = { w: [20, -20], b: -10 };

const NODES: Neuron[] = [
  { id: 'x1', label: 'x₁', x: 50, y: 60 },
  { id: 'x2', label: 'x₂', x: 50, y: 160 },
  { id: 'h1', label: 'OR', x: 180, y: 60 },
  { id: 'h2', label: 'AND', x: 180, y: 160 },
  { id: 'out', label: 'XOR', x: 310, y: 110 },
];
const EDGES: Edge[] = [
  { from: 'x1', to: 'h1', weight: HIDDEN[0].w[0] },
  { from: 'x2', to: 'h1', weight: HIDDEN[0].w[1] },
  { from: 'x1', to: 'h2', weight: HIDDEN[1].w[0] },
  { from: 'x2', to: 'h2', weight: HIDDEN[1].w[1] },
  { from: 'h1', to: 'out', weight: OUTPUT.w[0] },
  { from: 'h2', to: 'out', weight: OUTPUT.w[1] },
];
const CORNERS: [number, number][] = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
];

/** A two-layer network solving XOR: drag the inputs and watch each neuron switch on. */
@Component({
  selector: 'app-xor-network-figure',
  imports: [DecimalPipe, FigureFrame, FigureSlider],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <app-figure-frame
      heading="A network that computes XOR"
      hint="Pick a corner or drag the inputs. Brighter neurons are more switched on."
    >
      <svg
        viewBox="0 0 360 220"
        class="w-full max-w-md"
        role="img"
        [attr.aria-label]="description()"
      >
        @for (edge of edges; track $index) {
          <line
            [attr.x1]="node(edge.from).x"
            [attr.y1]="node(edge.from).y"
            [attr.x2]="node(edge.to).x"
            [attr.y2]="node(edge.to).y"
            [class]="edge.weight > 0 ? 'stroke-fig-1' : 'stroke-fig-negative'"
            [attr.stroke-width]="1 + 3 * signalOf(edge)"
            stroke-linecap="round"
          />
          <text
            [attr.x]="node(edge.from).x * 0.4 + node(edge.to).x * 0.6"
            [attr.y]="node(edge.from).y * 0.4 + node(edge.to).y * 0.6 - 6"
            class="fill-muted-foreground text-[9px]"
            text-anchor="middle"
          >
            {{ edge.weight > 0 ? '+' : '' }}{{ edge.weight }}
          </text>
        }
        @for (n of nodes; track n.id) {
          <circle
            [attr.cx]="n.x"
            [attr.cy]="n.y"
            r="24"
            class="fill-fig-1 stroke-border"
            [attr.fill-opacity]="0.08 + 0.8 * value()[n.id]"
            stroke-width="1.5"
          />
          <text
            [attr.x]="n.x"
            [attr.y]="n.y - 2"
            text-anchor="middle"
            class="fill-foreground text-[11px] font-semibold"
          >
            {{ n.label }}
          </text>
          <text
            [attr.x]="n.x"
            [attr.y]="n.y + 12"
            text-anchor="middle"
            class="fill-foreground text-[10px] tabular-nums"
          >
            {{ value()[n.id] | number: '1.2-2' }}
          </text>
        }
        <text x="180" y="210" text-anchor="middle" class="fill-muted-foreground text-[9px]">
          hidden layer: each neuron is σ(w·x + b)
        </text>
      </svg>

      <div class="mt-3 flex flex-wrap items-center gap-2" role="group" aria-label="Input corners">
        @for (corner of corners; track $index) {
          <button
            type="button"
            class="rounded-md border px-2.5 py-1 text-xs tabular-nums hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring"
            [class.bg-muted]="x1() === corner[0] && x2() === corner[1]"
            [attr.aria-pressed]="x1() === corner[0] && x2() === corner[1]"
            (click)="x1.set(corner[0]); x2.set(corner[1])"
          >
            ({{ corner[0] }}, {{ corner[1] }}) → {{ corner[0] !== corner[1] ? 1 : 0 }}
          </button>
        }
      </div>
      <div class="mt-3 grid gap-3 sm:grid-cols-2">
        <app-figure-slider
          label="x₁"
          [(value)]="x1"
          [step]="0.01"
          [readout]="(x1() | number: '1.2-2') ?? ''"
        />
        <app-figure-slider
          label="x₂"
          [(value)]="x2"
          [step]="0.01"
          [readout]="(x2() | number: '1.2-2') ?? ''"
        />
      </div>
      <p class="mt-3 font-mono text-xs leading-5 text-muted-foreground">
        OR&nbsp;: σ(20·{{ x1() | number: '1.2-2' }} + 20·{{ x2() | number: '1.2-2' }} − 10) =
        {{ value()['h1'] | number: '1.2-2' }}<br />
        AND: σ(20·{{ x1() | number: '1.2-2' }} + 20·{{ x2() | number: '1.2-2' }} − 30) =
        {{ value()['h2'] | number: '1.2-2' }}<br />
        XOR: σ(20·OR − 20·AND − 10) = {{ value()['out'] | number: '1.2-2' }}
      </p>
      <p caption>
        No single neuron can do this: one neuron draws one straight line, and XOR's classes sit on
        opposite corners. Here the <strong>hidden layer</strong> draws two lines (OR and AND), and
        the output neuron combines them. Blue edges add, red ones subtract; the thicker an edge, the
        more signal it carries right now.
      </p>
    </app-figure-frame>
  `,
})
export class XorNetworkFigure {
  protected readonly nodes = NODES;
  protected readonly edges = EDGES;
  protected readonly corners = CORNERS;
  protected readonly x1 = signal(0);
  protected readonly x2 = signal(1);

  protected readonly value = computed<Record<string, number>>(() => {
    const x = [this.x1(), this.x2()];
    const [or, and] = HIDDEN.map(({ w, b }) => sigmoid(w[0] * x[0] + w[1] * x[1] + b));
    const out = sigmoid(OUTPUT.w[0] * or + OUTPUT.w[1] * and + OUTPUT.b);
    return { x1: x[0], x2: x[1], h1: or, h2: and, out };
  });

  protected readonly description = computed(() => {
    const v = this.value();
    return (
      `Network diagram: inputs ${v['x1'].toFixed(2)} and ${v['x2'].toFixed(2)} feed an OR ` +
      `neuron (${v['h1'].toFixed(2)}) and an AND neuron (${v['h2'].toFixed(2)}); the output ` +
      `neuron gives XOR = ${v['out'].toFixed(2)}.`
    );
  });

  protected node(id: string): Neuron {
    return NODES.find((n) => n.id === id) as Neuron;
  }

  /** How much signal flows along an edge: the sending neuron's output, 0–1. */
  protected signalOf(edge: Edge): number {
    return this.value()[edge.from];
  }
}
