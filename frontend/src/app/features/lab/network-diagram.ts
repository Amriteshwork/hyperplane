import { DecimalPipe } from '@angular/common';
import { ChangeDetectionStrategy, Component, computed, input } from '@angular/core';

import { NetworkLayer } from '../../core/api/models/network-layer';
import { NetworkSummary } from '../../core/api/models/network-summary';
import { featureLabel } from './feature-label';

const WIDTH = 520;
const TOP = 44;
const GAP = 30;
const LEFT = 130;
const RIGHT = 100;
const LABEL_CHARS = 18;

interface Unit {
  x: number;
  y: number;
  label: string;
}

interface Column {
  x: number;
  heading: string;
  detail: string;
  units: Unit[];
  /** Units not drawn, shown as "+n more". */
  hidden: number;
}

interface Edge {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  positive: boolean;
  /** |weight| relative to the largest drawn between the same two layers, 0–1. */
  strength: number;
  title: string;
}

/** A trained network: every layer's size, and the learned weights among the units drawn. */
@Component({
  selector: 'app-network-diagram',
  imports: [DecimalPipe],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <figure class="grid gap-3">
      <div class="overflow-x-auto">
        <svg
          [attr.viewBox]="'0 0 ' + width + ' ' + height()"
          class="w-full max-w-2xl min-w-[480px]"
          role="img"
          [attr.aria-label]="description()"
        >
          @for (edge of edges(); track $index) {
            <line
              [attr.x1]="edge.x1"
              [attr.y1]="edge.y1"
              [attr.x2]="edge.x2"
              [attr.y2]="edge.y2"
              [class]="edge.positive ? 'stroke-fig-1' : 'stroke-fig-negative'"
              [attr.stroke-width]="0.4 + 2.6 * edge.strength"
              [attr.stroke-opacity]="0.15 + 0.75 * edge.strength"
            >
              <title>{{ edge.title }}</title>
            </line>
          }
          @for (column of columns(); track $index; let first = $first; let last = $last) {
            <text
              [attr.x]="column.x"
              y="14"
              text-anchor="middle"
              class="fill-foreground text-[11px] font-medium"
            >
              {{ column.heading }}
            </text>
            <text
              [attr.x]="column.x"
              y="28"
              text-anchor="middle"
              class="fill-muted-foreground text-[10px]"
            >
              {{ column.detail }}
            </text>
            @for (unit of column.units; track $index) {
              <circle
                [attr.cx]="column.x"
                [attr.cy]="unit.y"
                r="7"
                class="stroke-border"
                [class]="first ? 'fill-fig-3' : last ? 'fill-fig-2' : 'fill-fig-1'"
              >
                <title>{{ unit.label }}</title>
              </circle>
              @if (first || last) {
                <text
                  [attr.x]="column.x + (first ? -12 : 12)"
                  [attr.y]="unit.y + 3.5"
                  [attr.text-anchor]="first ? 'end' : 'start'"
                  class="fill-foreground text-[10px]"
                >
                  {{ short(unit.label) }}
                </text>
              }
            }
            @if (column.hidden) {
              <text
                [attr.x]="column.x"
                [attr.y]="column.units.at(-1)!.y + 22"
                text-anchor="middle"
                class="fill-muted-foreground text-[10px]"
              >
                +{{ column.hidden | number }} more
              </text>
            }
          }
        </svg>
      </div>
      <figcaption class="max-w-prose text-xs text-muted-foreground">
        Blue lines are positive weights, red ones negative; thicker means larger (compared within
        each pair of layers). Hover a line for its value. Wide layers show their first neurons, and
        the inputs the first layer weighs most.
      </figcaption>
    </figure>
  `,
})
export class NetworkDiagram {
  readonly network = input.required<NetworkSummary>();

  protected readonly width = WIDTH;

  protected readonly height = computed(() => {
    const tallest = Math.max(...this.network().layers.map((l) => l.shown.length));
    const more = this.network().layers.some((l) => l.units > l.shown.length);
    return TOP + (tallest - 1) * GAP + (more ? 36 : 16);
  });

  protected readonly columns = computed<Column[]>(() => {
    const layers = this.network().layers;
    const tallest = Math.max(...layers.map((l) => l.shown.length));
    const step = (WIDTH - LEFT - RIGHT) / Math.max(1, layers.length - 1);
    let hiddenIndex = 0;
    return layers.map((layer, i) => {
      // Centre shorter columns on the tallest one.
      const offset = TOP + ((tallest - layer.shown.length) * GAP) / 2;
      const x = LEFT + i * step;
      return {
        x,
        heading: layer.kind === 'hidden' ? `hidden ${++hiddenIndex}` : `${layer.kind}s`,
        detail: detail(layer),
        units: layer.shown.map((label, k) => ({
          x,
          y: offset + k * GAP,
          label: layer.kind === 'input' ? featureLabel(label) : label,
        })),
        hidden: layer.units - layer.shown.length,
      };
    });
  });

  protected readonly edges = computed<Edge[]>(() => {
    const columns = this.columns();
    return this.network().weights.flatMap((matrix, i) => {
      const largest = Math.max(1e-12, ...matrix.flat().map(Math.abs));
      return matrix.flatMap((row, a) =>
        row.map((weight, b) => {
          const from = columns[i].units[a];
          const to = columns[i + 1].units[b];
          return {
            x1: from.x,
            y1: from.y,
            x2: to.x,
            y2: to.y,
            positive: weight >= 0,
            strength: Math.abs(weight) / largest,
            title: `${from.label} → ${to.label}: ${weight > 0 ? '+' : ''}${weight.toPrecision(3)}`,
          };
        }),
      );
    });
  });

  protected readonly description = computed(() => {
    const layers = this.network()
      .layers.map((l) => `${l.units} ${l.kind}`)
      .join(', ');
    return (
      `Diagram of the trained network: ${layers} units, ` +
      `${this.network().parameters} weights and biases. Line colour and thickness show each weight.`
    );
  });

  protected short(label: string): string {
    return label.length > LABEL_CHARS ? label.slice(0, LABEL_CHARS - 1) + '…' : label;
  }
}

function detail(layer: NetworkLayer): string {
  return layer.activation ? `${layer.units} · ${layer.activation}` : `${layer.units}`;
}
