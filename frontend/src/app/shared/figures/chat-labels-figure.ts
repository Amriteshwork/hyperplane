import { ChangeDetectionStrategy, Component, computed, signal } from '@angular/core';

import { FigureFrame } from './figure-frame';

type Speaker = 'marker' | 'user' | 'assistant';

const STOP = '<stop>';

const CHAT: readonly { text: string; speaker: Speaker }[] = [
  { text: '<user>', speaker: 'marker' },
  { text: 'what', speaker: 'user' },
  { text: 'does', speaker: 'user' },
  { text: 'zoe', speaker: 'user' },
  { text: 'eat', speaker: 'user' },
  { text: '?', speaker: 'user' },
  { text: '<assistant>', speaker: 'marker' },
  { text: 'jam', speaker: 'assistant' },
  { text: '.', speaker: 'assistant' },
  { text: STOP, speaker: 'assistant' },
  { text: '<user>', speaker: 'marker' },
  { text: 'and', speaker: 'user' },
  { text: 'raj', speaker: 'user' },
  { text: '?', speaker: 'user' },
  { text: '<assistant>', speaker: 'marker' },
  { text: 'figs', speaker: 'assistant' },
  { text: '.', speaker: 'assistant' },
  { text: STOP, speaker: 'assistant' },
];

/** Step along a chat: at each position, what's the model predicting, and does it count? */
@Component({
  selector: 'app-chat-labels-figure',
  imports: [FigureFrame],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <app-figure-frame heading="What the loss sees" hint="Click a token, or switch the view.">
      <div class="flex flex-wrap gap-1" role="group" aria-label="The chat as one sequence">
        @for (token of chat; track $index; let i = $index) {
          <button
            type="button"
            class="rounded-md border px-1.5 py-1 font-mono text-xs focus-visible:outline-2 focus-visible:outline-ring"
            [class]="tokenClass(i)"
            [attr.aria-pressed]="at() === i"
            (click)="at.set(i)"
          >
            {{ token.text }}
          </button>
        }
      </div>
      <button
        type="button"
        class="mt-2 rounded-md border px-3 py-1.5 text-xs hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring"
        [attr.aria-pressed]="masked()"
        (click)="masked.set(!masked())"
      >
        {{ masked() ? 'Show the speakers' : 'Show what counts towards the loss' }}
      </button>

      <div class="mt-3 rounded-md bg-muted/50 p-3 text-sm" aria-live="polite">
        @if (next(); as target) {
          <p>
            Reading up to <span class="font-mono">{{ chat[at()].text }}</span
            >, the model predicts <span class="font-mono font-semibold">{{ target.text }}</span
            >.
          </p>
          <p class="mt-1">
            <span class="font-semibold" [class.text-fig-3]="target.speaker === 'assistant'">{{
              target.speaker === 'assistant' ? 'This counts.' : 'Label −100, so it costs nothing.'
            }}</span>
            {{ reason() }}
          </p>
        } @else {
          <p>This is the last token, so there's nothing left to predict.</p>
        }
      </div>
      <p caption>
        A conversation is one sequence with one kind of position, predicting the next token. Only
        the positions whose next token belongs to the assistant add to the loss. {{ counted }} of
        {{ chat.length - 1 }} count here, and the others add nothing.
      </p>
    </app-figure-frame>
  `,
})
export class ChatLabelsFigure {
  protected readonly chat = CHAT;
  protected readonly counted = CHAT.slice(1).filter((t) => t.speaker === 'assistant').length;
  protected readonly at = signal(6);
  protected readonly masked = signal(false);
  protected readonly next = computed(() => this.chat[this.at() + 1] ?? null);

  protected readonly reason = computed(() => {
    const target = this.next();
    if (target?.speaker === 'assistant')
      return target.text === STOP
        ? 'This is how it learns to finish.'
        : 'The assistant should say this.';
    return target?.speaker === 'user'
      ? "It's never taught to write the user's words."
      : 'The template adds this marker, so there is nothing to learn.';
  });

  protected tokenClass(i: number): string {
    const speaker = this.chat[i].speaker;
    const selected = this.at() === i ? ' ring-2 ring-ring' : '';
    if (this.masked())
      return (speaker === 'assistant' ? 'border-fig-3 bg-fig-3/20' : 'opacity-40') + selected;
    const tint: Record<Speaker, string> = {
      marker: 'bg-muted',
      user: 'border-fig-1 bg-fig-1/15',
      assistant: 'border-fig-3 bg-fig-3/20',
    };
    return tint[speaker] + selected;
  }
}
