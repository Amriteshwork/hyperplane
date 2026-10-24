import { DecimalPipe } from '@angular/common';
import {
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  computed,
  inject,
  signal,
} from '@angular/core';

import { FigureFrame } from './figure-frame';
import { FigureSlider } from './figure-slider';
import { linear, path, range } from './lib/plot';
import { normal, seeded, uniform } from './lib/random';
import { Adam, createNetwork, gradients, loss, predict } from './lib/tiny-mlp';

const EPOCHS = 1000;
/** A snapshot of the fitted curve every this many epochs (the slider's step). */
const EVERY = 5;
const PER_FRAME = 10;
const W = 300;
const H = 190;
const GRID = range(-3, 3, 80);
const fx = linear(-3, 3, 10, W - 10);
const fy = linear(-2, 2, H - 10, 10);
const ex = linear(0, EPOCHS, 30, W - 10);
const ey = (mse: number) => linear(Math.log10(0.01), Math.log10(1.5), H - 20, 10)(Math.log10(mse));

interface Sample {
  X: number[][];
  y: number[];
}

/** Same set-up as the early-stopping page's code: 30 noisy rows, a 24×24 tanh network, Adam. */
function noisySine(rng: () => number, n: number): Sample {
  const X: number[][] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const v = uniform(rng, -3, 3);
    X.push([v]);
    y.push(Math.sin(v) + 0.3 * normal(rng));
  }
  return { X, y };
}

/** Trains a network live, then lets the reader scrub through epochs to see overfitting happen. */
@Component({
  selector: 'app-early-stopping-figure',
  imports: [DecimalPipe, FigureFrame, FigureSlider],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <app-figure-frame
      heading="Watch a network overfit, and where early stopping would stop it"
      hint="The network trains live (1,000 epochs). Then slide through the epochs."
    >
      <div class="grid gap-3 sm:grid-cols-2">
        <svg
          [attr.viewBox]="'0 0 ' + w + ' ' + h"
          class="w-full"
          role="img"
          [attr.aria-label]="fitDescription()"
        >
          <path
            [attr.d]="truth"
            class="fill-none stroke-muted-foreground"
            stroke-width="1.5"
            stroke-dasharray="4 4"
          />
          @for (p of valPoints; track $index) {
            <circle
              [attr.cx]="p[0]"
              [attr.cy]="p[1]"
              r="1.6"
              class="fill-muted-foreground"
              fill-opacity="0.35"
            />
          }
          @for (p of trainPoints; track $index) {
            <circle [attr.cx]="p[0]" [attr.cy]="p[1]" r="3" class="fill-fig-1" />
          }
          <path [attr.d]="fit()" class="fill-none stroke-fig-2" stroke-width="2.5" />
          <text x="12" y="16" class="fill-muted-foreground text-[9px]">
            the fit at epoch {{ epoch() }} (orange) · training rows (blue)
          </text>
        </svg>
        <svg
          [attr.viewBox]="'0 0 ' + w + ' ' + h"
          class="w-full"
          role="img"
          [attr.aria-label]="lossDescription()"
        >
          <path [attr.d]="curve(trainLoss())" class="fill-none stroke-fig-1" stroke-width="2" />
          <path [attr.d]="curve(valLoss())" class="fill-none stroke-fig-2" stroke-width="2" />
          <line
            [attr.x1]="ex(epoch())"
            [attr.x2]="ex(epoch())"
            y1="10"
            [attr.y2]="h - 20"
            class="stroke-foreground"
            stroke-dasharray="3 3"
          />
          @if (best(); as b) {
            <circle [attr.cx]="ex(b.epoch)" [attr.cy]="ey(b.loss)" r="4.5" class="fill-fig-3" />
          }
          <text x="34" y="20" class="fill-fig-1 text-[9px]">training loss</text>
          <text x="34" y="32" class="fill-fig-2 text-[9px]">validation loss</text>
          <text
            [attr.x]="w - 10"
            [attr.y]="h - 6"
            text-anchor="end"
            class="fill-muted-foreground text-[9px]"
          >
            epoch →
          </text>
          <text x="4" y="14" class="fill-muted-foreground text-[8px]">MSE (log)</text>
        </svg>
      </div>

      @if (done() < epochs) {
        <p class="mt-2 text-xs text-muted-foreground" aria-live="polite">
          Training… epoch {{ done() }} of {{ epochs }}
        </p>
      }
      <app-figure-slider
        class="mt-2"
        label="Epoch"
        [(value)]="epoch"
        [min]="0"
        [max]="done()"
        [step]="every"
      />
      <div class="mt-2 flex flex-wrap items-center gap-2 text-xs">
        <button
          type="button"
          class="rounded-md border px-2.5 py-1 hover:bg-muted disabled:opacity-50"
          [disabled]="!best()"
          (click)="epoch.set(best()?.snapshot ?? 0)"
        >
          Jump to the best epoch
        </button>
        <button
          type="button"
          class="rounded-md border px-2.5 py-1 hover:bg-muted"
          (click)="epoch.set(done())"
        >
          Jump to the end
        </button>
        <span class="ms-auto tabular-nums text-muted-foreground">
          training MSE
          <span class="font-mono text-foreground">{{
            trainLoss()[epoch()] | number: '1.3-3'
          }}</span>
          · validation MSE
          <span class="font-mono text-foreground">{{ valLoss()[epoch()] | number: '1.3-3' }}</span>
        </span>
      </div>
      <p caption>
        The blue dots are the only 30 rows the network trains on; the faint ones are held out. The
        training loss falls all the way down, but the validation loss bottoms out early (green dot)
        and then climbs: past that point the orange curve bends to chase the noise in its 30 rows,
        not the true shape (dashed). <strong>Early stopping</strong> watches the validation loss,
        stops a few epochs after its best, and keeps the best weights.
      </p>
    </app-figure-frame>
  `,
})
export class EarlyStoppingFigure {
  protected readonly w = W;
  protected readonly h = H;
  protected readonly epochs = EPOCHS;
  protected readonly every = EVERY;
  protected readonly ex = ex;
  protected readonly ey = ey;
  protected readonly truth = path(GRID.map((v) => [fx(v), fy(Math.sin(v))] as const));

  private readonly rng = seeded(1);
  private readonly train = noisySine(this.rng, 30);
  private readonly val = noisySine(this.rng, 100);
  private readonly net = createNetwork([1, 24, 24, 1], 'tanh', 'identity', this.rng);
  private readonly adam = new Adam(this.net, 0.02);

  protected readonly trainPoints = this.train.X.map((x, i) => [fx(x[0]), fy(this.train.y[i])]);
  protected readonly valPoints = this.val.X.map((x, i) => [fx(x[0]), fy(this.val.y[i])]);

  protected readonly done = signal(0);
  protected readonly epoch = signal(0);
  protected readonly trainLoss = signal<number[]>([loss(this.net, this.train.X, this.train.y)]);
  protected readonly valLoss = signal<number[]>([loss(this.net, this.val.X, this.val.y)]);
  private readonly snapshots = signal<string[]>([this.curveOfNetwork()]);

  protected readonly fit = computed(() => this.snapshots()[Math.round(this.epoch() / EVERY)] ?? '');

  /** The epoch with the lowest validation loss, and the nearest snapshot to show it. */
  protected readonly best = computed(() => {
    const val = this.valLoss();
    if (val.length < 2) return null;
    const epoch = val.indexOf(Math.min(...val));
    return { epoch, loss: val[epoch], snapshot: Math.round(epoch / EVERY) * EVERY };
  });

  protected readonly fitDescription = computed(
    () =>
      `Scatter of 30 training rows on a noisy sine wave, with the network's fit at epoch ${this.epoch()}.`,
  );
  protected readonly lossDescription = computed(() => {
    const b = this.best();
    return b
      ? `Training and validation loss per epoch; the validation loss is lowest (${b.loss.toFixed(3)}) at epoch ${b.epoch}.`
      : 'Training and validation loss per epoch.';
  });

  constructor() {
    let frame = 0;
    const run = () => {
      if (this.trainSome()) frame = requestAnimationFrame(run);
    };
    frame = requestAnimationFrame(run);
    inject(DestroyRef).onDestroy(() => cancelAnimationFrame(frame));
  }

  protected curve(values: number[]): string {
    return path(values.map((v, i) => [ex(i), ey(Math.max(v, 0.01))] as const));
  }

  /** A few epochs per animation frame; returns whether there is more to do. */
  private trainSome(): boolean {
    const start = this.done();
    if (start >= EPOCHS) return false;
    const trainLoss = [...this.trainLoss()];
    const valLoss = [...this.valLoss()];
    const snapshots = [...this.snapshots()];
    const end = Math.min(start + PER_FRAME, EPOCHS);
    for (let e = start + 1; e <= end; e++) {
      this.adam.step(this.net, gradients(this.net, this.train.X, this.train.y));
      trainLoss.push(loss(this.net, this.train.X, this.train.y));
      valLoss.push(loss(this.net, this.val.X, this.val.y));
      if (e % EVERY === 0) snapshots.push(this.curveOfNetwork());
    }
    const following = this.epoch() === start || this.epoch() === Math.floor(start / EVERY) * EVERY;
    this.trainLoss.set(trainLoss);
    this.valLoss.set(valLoss);
    this.snapshots.set(snapshots);
    this.done.set(end);
    if (following) this.epoch.set(Math.floor(end / EVERY) * EVERY); // follow training live
    return end < EPOCHS;
  }

  private curveOfNetwork(): string {
    const predictions = predict(
      this.net,
      GRID.map((v) => [v]),
    );
    return path(
      GRID.map((v, i) => [fx(v), fy(Math.max(-2, Math.min(2, predictions[i])))] as const),
    );
  }
}
