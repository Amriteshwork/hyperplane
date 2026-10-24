import { DecimalPipe } from '@angular/common';
import { ChangeDetectionStrategy, Component, computed, signal } from '@angular/core';

import { FigureFrame } from './figure-frame';
import { FigureSlider } from './figure-slider';
import { linear, path, range } from './lib/plot';
import { A, Point, adam, gradientDescent, lossAt, momentum } from './lib/ravine';

const STEPS = 100;
const W = 340;
const H = 220;
const px = linear(-1.1, 1.1, 10, W - 10);
const py = linear(-0.3, 1.15, H - 10, 10);

const RUNNERS = [
  {
    name: 'Gradient descent',
    css: 'stroke-fig-negative',
    dot: 'fill-fig-negative',
    points: gradientDescent(STEPS),
  },
  { name: 'Momentum', css: 'stroke-fig-4', dot: 'fill-fig-4', points: momentum(STEPS) },
  { name: 'Adam', css: 'stroke-fig-1', dot: 'fill-fig-1', points: adam(STEPS) },
];

/** Three optimisers race down the same ravine-shaped loss; slide through the steps. */
@Component({
  selector: 'app-optimizer-race-figure',
  imports: [DecimalPipe, FigureFrame, FigureSlider],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <app-figure-frame
      heading="Three optimisers, one ravine"
      hint="Slide through the steps. The minimum is the cross; contours are equal loss."
    >
      <svg
        [attr.viewBox]="'0 0 ' + w + ' ' + h"
        class="w-full"
        role="img"
        [attr.aria-label]="description()"
      >
        @for (c of contours; track $index) {
          <ellipse
            [attr.cx]="px(0)"
            [attr.cy]="py(0)"
            [attr.rx]="c.rx"
            [attr.ry]="c.ry"
            class="fill-none stroke-border"
          />
        }
        <path [attr.d]="cross" class="stroke-foreground" stroke-width="1.5" />
        @for (runner of runners; track runner.name) {
          <path
            [attr.d]="trail(runner.points)"
            [class]="'fill-none ' + runner.css"
            stroke-width="1.8"
            stroke-linejoin="round"
          />
          <circle
            [attr.cx]="px(runner.points[step()][0])"
            [attr.cy]="py(runner.points[step()][1])"
            r="4.5"
            [class]="runner.dot"
          />
        }
        <text [attr.x]="px(0) + 6" [attr.y]="h - 14" class="fill-muted-foreground text-[9px]">
          steep: w₀ →
        </text>
        <text [attr.x]="px(-1.05)" [attr.y]="py(0.6)" class="fill-muted-foreground text-[9px]">
          flat: w₁ ↑
        </text>
      </svg>
      <app-figure-slider class="mt-2" label="Step" [(value)]="step" [min]="0" [max]="steps" />
      <ul class="mt-3 grid list-none gap-1 ps-0 text-xs sm:grid-cols-3">
        @for (runner of runners; track runner.name) {
          <li class="flex items-center gap-2">
            <svg width="14" height="4" aria-hidden="true">
              <line x1="0" y1="2" x2="14" y2="2" [class]="runner.css" stroke-width="3" />
            </svg>
            {{ runner.name }}:
            <span class="font-mono tabular-nums">{{
              loss(runner.points[step()]) | number: '1.4-4'
            }}</span>
          </li>
        }
      </ul>
      <p caption>
        The loss is 100 times steeper across the ravine than along it.
        <strong>Gradient descent</strong> needs a step small enough not to fly off the steep walls,
        so it bounces across and crawls along the floor. <strong>Momentum</strong> averages past
        gradients: the bounces cancel and the slow direction builds up speed.
        <strong>Adam</strong> also divides each direction by the size of its own gradients, so both
        directions shrink at the same pace. Real networks have millions of directions like these,
        which is why Adam is the default optimiser.
      </p>
    </app-figure-frame>
  `,
})
export class OptimizerRaceFigure {
  protected readonly w = W;
  protected readonly h = H;
  protected readonly px = px;
  protected readonly py = py;
  protected readonly steps = STEPS;
  protected readonly runners = RUNNERS;
  protected readonly loss = lossAt;
  protected readonly step = signal(25);
  protected readonly cross = `M${px(0) - 5},${py(0)}h10M${px(0)},${py(0) - 5}v10`;

  /** Ellipses of equal loss: rx and ry in pixels for loss levels spaced like a geometric series. */
  protected readonly contours = range(0, 5, 6).map((i) => {
    const level = 0.6 * 0.5 ** i;
    const scaleX = px(1) - px(0);
    const scaleY = py(0) - py(1);
    return { rx: Math.sqrt((2 * level) / A) * scaleX, ry: Math.sqrt(2 * level) * scaleY };
  });

  protected trail(points: Point[]): string {
    return path(points.slice(0, this.step() + 1).map(([a, b]) => [px(a), py(b)] as const));
  }

  protected readonly description = computed(() => {
    const at = RUNNERS.map((r) => `${r.name} ${lossAt(r.points[this.step()]).toFixed(4)}`);
    return `Contour plot of a ravine-shaped loss after ${this.step()} steps; losses: ${at.join(', ')}.`;
  });
}
