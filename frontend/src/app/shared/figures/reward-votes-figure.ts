import { ChangeDetectionStrategy, Component, computed, signal } from '@angular/core';

import { FigureFrame } from './figure-frame';
import { REPLIES, fitReward, rewards } from './lib/rlhf';

/** Only replies raters really compare; the last reply in REPLIES is deliberately left out. */
const MATCHUPS: readonly (readonly [number, number])[] = [
  [3, 2],
  [2, 0],
  [4, 3],
  [3, 1],
  [2, 1],
  [4, 2],
];

/** The reader votes between replies, and the reward model refits after every vote. */
@Component({
  selector: 'app-reward-votes-figure',
  imports: [FigureFrame],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <app-figure-frame
      heading="You be the rater"
      hint="The question was “what does zoe eat?”, and she eats jam. Pick the better reply."
    >
      @if (matchup(); as pair) {
        <div class="grid gap-2 sm:grid-cols-2" role="group" aria-label="Which reply is better?">
          @for (reply of pair; track reply) {
            <button
              type="button"
              class="rounded-md border p-3 text-sm hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring"
              (click)="vote(reply, reply === pair[0] ? pair[1] : pair[0])"
            >
              “{{ replies[reply] }}”
            </button>
          }
        </div>
      } @else {
        <p class="rounded-md bg-muted/50 p-3 text-sm">
          That's all {{ matchups.length }} votes. Try voting differently: start over and always pick
          the blunter reply.
        </p>
      }
      <button
        type="button"
        class="mt-2 rounded-md border px-3 py-1.5 text-xs hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring"
        (click)="votes.set([])"
      >
        Start over
      </button>

      <p class="mt-3 text-xs font-semibold">
        What the reward model has learnt from {{ votes().length }}
        {{ votes().length === 1 ? 'vote' : 'votes' }}
      </p>
      <dl class="mt-1 grid grid-cols-3 gap-2 text-center text-xs" aria-live="polite">
        @for (feature of features; track feature; let i = $index) {
          <div class="rounded-md bg-muted/50 p-2">
            <dt class="text-muted-foreground">{{ feature }}</dt>
            <dd class="text-base font-semibold tabular-nums">{{ signed(weights()[i]) }}</dd>
          </div>
        }
      </dl>
      <ul class="list-none pl-0 mt-3 space-y-1 text-xs">
        @for (row of scored(); track row.reply) {
          <li class="grid grid-cols-[1fr_2.5rem] gap-2">
            <span class="truncate">{{ row.reply }}</span>
            <span class="text-right font-semibold tabular-nums">{{ signed(row.score) }}</span>
          </li>
        }
      </ul>
      <p caption>
        Every vote adds one training example, and the weights are refit from scratch. Look at the
        last reply in the list: you were never asked about it, yet it already has a score, because
        it has the same features as replies you liked, only more of them. That untested score is
        what goes wrong in the next figure.
      </p>
    </app-figure-frame>
  `,
})
export class RewardVotesFigure {
  protected readonly replies = REPLIES;
  protected readonly matchups = MATCHUPS;
  protected readonly features = ['right', 'polite', 'extra words'];
  protected readonly votes = signal<readonly (readonly [number, number])[]>([]);
  protected readonly matchup = computed(() => this.matchups[this.votes().length] ?? null);
  protected readonly weights = computed(() => fitReward(this.votes()));
  protected readonly scored = computed(() => {
    const scores = rewards(this.weights());
    return REPLIES.map((reply, i) => ({ reply, score: scores[i] })).sort(
      (a, b) => b.score - a.score,
    );
  });

  protected vote(winner: number, loser: number): void {
    this.votes.update((votes) => [...votes, [winner, loser]]);
  }

  protected signed(value: number): string {
    return `${value >= 0 ? '+' : '−'}${Math.abs(value).toFixed(1)}`;
  }
}
