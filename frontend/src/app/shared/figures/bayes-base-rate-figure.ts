import { ChangeDetectionStrategy, Component, computed, signal } from '@angular/core';

import { FigureFrame } from './figure-frame';
import { FigureSlider } from './figure-slider';

const WIDTH = 460;
const HEIGHT = 170;
const PEOPLE = 10_000;
const MOSAIC = 300;
const BAR_X = 340;

/** Bayes' theorem as areas: how the base rate decides what a positive test means. */
@Component({
  selector: 'app-bayes-base-rate-figure',
  imports: [FigureFrame, FigureSlider],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <app-figure-frame
      heading="What a positive test means"
      hint="Change how common the condition is."
    >
      <svg
        [attr.viewBox]="'0 0 ' + width + ' ' + height"
        class="w-full rounded-md border"
        role="img"
        [attr.aria-label]="description()"
      >
        <text x="0" y="12" class="fill-muted-foreground text-[10px]">Everyone, by condition</text>
        <rect
          x="0"
          y="20"
          [attr.width]="sickWidth()"
          [attr.height]="box"
          class="fill-fig-1"
          fill-opacity="0.2"
        />
        <rect
          x="0"
          y="20"
          [attr.width]="sickWidth()"
          [attr.height]="(box * hit()) / 100"
          class="fill-fig-1"
        />
        <rect
          [attr.x]="sickWidth()"
          y="20"
          [attr.width]="mosaic - sickWidth()"
          [attr.height]="box"
          class="fill-fig-2"
          fill-opacity="0.2"
        />
        <rect
          [attr.x]="sickWidth()"
          y="20"
          [attr.width]="mosaic - sickWidth()"
          [attr.height]="(box * falseAlarm()) / 100"
          class="fill-fig-2"
        />
        <text [attr.x]="barX" y="12" class="fill-muted-foreground text-[10px]">Positive tests</text>
        <rect [attr.x]="barX" y="20" width="60" [attr.height]="box * share()" class="fill-fig-1" />
        <rect
          [attr.x]="barX"
          [attr.y]="20 + box * share()"
          width="60"
          [attr.height]="box * (1 - share())"
          class="fill-fig-2"
        />
      </svg>
      <div class="mt-3 grid gap-3 sm:grid-cols-3">
        <app-figure-slider
          label="How common (%)"
          [(value)]="prior"
          [min]="0.1"
          [max]="50"
          [step]="0.1"
          [readout]="prior() + '%'"
        />
        <app-figure-slider
          label="Caught if sick (%)"
          [(value)]="hit"
          [min]="50"
          [max]="100"
          [readout]="hit() + '%'"
        />
        <app-figure-slider
          label="Falsely flagged if healthy (%)"
          [(value)]="falseAlarm"
          [min]="0"
          [max]="30"
          [step]="0.5"
          [readout]="falseAlarm() + '%'"
        />
      </div>
      <p class="mt-2 text-sm" aria-live="polite">
        Of {{ people.toLocaleString('en') }} people,
        {{ truePositives().toLocaleString('en', { maximumFractionDigits: 0 }) }} sick and
        {{ falsePositives().toLocaleString('en', { maximumFractionDigits: 0 }) }} healthy people
        test positive, so a positive means a
        <span class="font-mono tabular-nums" data-testid="posterior">{{ percent() }}</span
        >% chance of being sick.
      </p>
      <p caption>
        On the left, each column's width is how common that group is (the prior), and the solid part
        is its share testing positive (the likelihood). On the right, the two solid parts are
        stacked: the blue share is the posterior. With a rare condition, a small share of a huge
        group outnumbers a large share of a tiny one.
      </p>
    </app-figure-frame>
  `,
})
export class BayesBaseRateFigure {
  protected readonly width = WIDTH;
  protected readonly height = HEIGHT;
  protected readonly mosaic = MOSAIC;
  protected readonly barX = BAR_X;
  protected readonly box = HEIGHT - 28;
  protected readonly people = PEOPLE;

  protected readonly prior = signal(1);
  protected readonly hit = signal(90);
  protected readonly falseAlarm = signal(5);

  protected readonly sickWidth = computed(() => (MOSAIC * this.prior()) / 100);
  protected readonly truePositives = computed(() => (PEOPLE * this.prior() * this.hit()) / 10_000);
  protected readonly falsePositives = computed(
    () => (PEOPLE * (100 - this.prior()) * this.falseAlarm()) / 10_000,
  );
  protected readonly share = computed(() => {
    const positives = this.truePositives() + this.falsePositives();
    return positives ? this.truePositives() / positives : 0;
  });
  protected readonly percent = computed(() => (100 * this.share()).toFixed(1));
  protected readonly description = computed(
    () =>
      `With ${this.prior()}% sick, ${this.hit()}% caught and ${this.falseAlarm()}% falsely flagged, ` +
      `a positive test means a ${this.percent()}% chance of being sick.`,
  );
}
