import { ChangeDetectionStrategy, Component, computed, signal } from '@angular/core';

import { FigureFrame } from './figure-frame';
import { CHAT_RUNS } from './lib/measured';

const PROMPT = 'kim eats eggs.<u>what does kim eat?<a>';

/** The same prompt through each model, with what each really scored on unseen chats. */
@Component({
  selector: 'app-base-vs-chat-figure',
  imports: [FigureFrame],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <app-figure-frame
      heading="Before and after fine-tuning, measured"
      hint="Click a model. All three see the same prompt."
    >
      <p class="break-all rounded-md bg-muted/50 p-2 font-mono text-xs">{{ prompt }}</p>

      <div class="mt-3 grid gap-2 sm:grid-cols-3" role="group" aria-label="Model">
        @for (run of runs; track run.label; let i = $index) {
          <button
            type="button"
            class="rounded-md border p-2 text-left hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring"
            [class.border-fig-1]="at() === i"
            [class.bg-fig-1/10]="at() === i"
            [attr.aria-pressed]="at() === i"
            (click)="at.set(i)"
          >
            <span class="block text-xs">{{ run.label }}</span>
            <span class="block text-xl font-semibold tabular-nums">{{ percent(run.rate) }}</span>
            <span class="mt-1 block h-1.5 rounded bg-muted">
              <span class="block h-1.5 rounded bg-fig-1" [style.width.%]="run.rate * 100"></span>
            </span>
          </button>
        }
      </div>

      <div class="mt-3 rounded-md bg-muted/50 p-3 text-sm" aria-live="polite">
        @if (run().reply !== null) {
          <p>
            It writes:
            <span class="font-mono font-semibold whitespace-pre">“{{ run().reply }}”</span>
          </p>
        }
        <p [class.mt-1]="run().reply !== null">{{ run().note }}</p>
      </div>
      <p caption>
        Each percentage is the share of 50 conversations it never saw that it got exactly right,
        stop token included. <span class="font-mono">|</span> is the stop token. All of this comes
        from running this page's PyTorch code, and the numbers change a little with the seed.
      </p>
    </app-figure-frame>
  `,
})
export class BaseVsChatFigure {
  protected readonly prompt = PROMPT;
  protected readonly runs = CHAT_RUNS;
  protected readonly at = signal(2);
  protected readonly run = computed(() => this.runs[this.at()]);

  protected percent(value: number): string {
    return `${Math.round(value * 100)}%`;
  }
}
