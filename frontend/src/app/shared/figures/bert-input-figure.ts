import { ChangeDetectionStrategy, Component, computed, signal } from '@angular/core';

import { FigureFrame } from './figure-frame';
import { packPair } from './lib/bert';

/** Fits a phone screen; real BERT allows 512. */
const LIMIT = 16;

function textValue(event: Event): string {
  return (event.target as HTMLInputElement).value;
}

/** BERT's input: two sentences packed as [CLS] A [SEP] B [SEP], each token a sum of three embeddings. */
@Component({
  selector: 'app-bert-input-figure',
  imports: [FigureFrame],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <app-figure-frame
      heading="Token + segment + position"
      hint="Type two sentences, then pick a token to see what its input vector is made of."
    >
      <div class="grid gap-2 sm:grid-cols-2">
        <label class="block text-xs">
          <span class="text-fig-3">Sentence A</span>
          <input
            type="text"
            class="mt-1 block w-full rounded-md border bg-background px-2 py-1.5 text-sm"
            [value]="a()"
            (input)="a.set(read($event))"
          />
        </label>
        <label class="block text-xs">
          <span class="text-fig-2">Sentence B (optional)</span>
          <input
            type="text"
            class="mt-1 block w-full rounded-md border bg-background px-2 py-1.5 text-sm"
            [value]="b()"
            (input)="b.set(read($event))"
          />
        </label>
      </div>

      <div class="mt-3 overflow-x-auto pb-1">
        <table class="border-separate border-spacing-1 text-center font-mono text-xs">
          <caption class="sr-only">
            Each column is one input token; rows are the three embeddings added together.
          </caption>
          <tbody>
            <tr>
              <th scope="row" class="pr-1 text-left font-sans font-normal text-muted-foreground">
                token
              </th>
              @for (t of tokens(); track $index; let i = $index) {
                <td class="p-0">
                  <button
                    type="button"
                    class="min-w-11 rounded border border-fig-1/60 bg-fig-1/10 px-1.5 py-1 focus-visible:outline-2 focus-visible:outline-ring"
                    [class.ring-2]="selected() === i"
                    [class.ring-fig-1]="selected() === i"
                    [attr.aria-pressed]="selected() === i"
                    (click)="selected.set(i)"
                  >
                    {{ t.token }}
                  </button>
                </td>
              }
            </tr>
            <tr>
              <th scope="row" class="pr-1 text-left font-sans font-normal text-muted-foreground">
                segment
              </th>
              @for (t of tokens(); track $index) {
                <td
                  class="rounded border px-1.5 py-1"
                  [class]="
                    t.segment === 0 ? 'border-fig-3/60 bg-fig-3/15' : 'border-fig-2/60 bg-fig-2/15'
                  "
                >
                  {{ t.segment === 0 ? 'A' : 'B' }}
                </td>
              }
            </tr>
            <tr>
              <th scope="row" class="pr-1 text-left font-sans font-normal text-muted-foreground">
                position
              </th>
              @for (t of tokens(); track $index) {
                <td class="rounded border border-fig-4/60 bg-fig-4/15 px-1.5 py-1">
                  {{ t.position }}
                </td>
              }
            </tr>
          </tbody>
        </table>
      </div>

      <div class="mt-2 rounded-md bg-muted/50 p-3 text-sm" aria-live="polite">
        @if (current(); as t) {
          <p class="font-mono">
            x<sub>{{ t.position }}</sub> = E<sub>token</sub>[{{ t.token }}] + E<sub>segment</sub>[{{
              t.segment === 0 ? 'A' : 'B'
            }}] + E<sub>position</sub>[{{ t.position }}]
          </p>
          <p class="mt-1 text-xs text-muted-foreground">{{ explain(t.token) }}</p>
        }
      </div>
      <p caption>
        Every token's input is the sum of three learned vectors, all the same length (768 in
        BERT-base). The token row says what the word is, the segment row which sentence it belongs
        to, and the position row where it stands. A word that appears in both sentences gets the
        same token vector but a different segment and position, so attention can still tell the two
        apart. Words ending in -ing or -ed are split the way WordPiece splits rarer words: "playing"
        becomes "play" and "##ing". This figure keeps {{ limit }} tokens; BERT keeps up to 512.
      </p>
    </app-figure-frame>
  `,
})
export class BertInputFigure {
  protected readonly limit = LIMIT;
  protected readonly read = textValue;
  protected readonly a = signal('where is the cat?');
  protected readonly b = signal('it is sleeping on the mat.');
  protected readonly selected = signal(0);

  protected readonly tokens = computed(() => packPair(this.a(), this.b(), LIMIT));
  protected readonly current = computed(() => {
    const tokens = this.tokens();
    return tokens[Math.min(this.selected(), tokens.length - 1)];
  });

  protected explain(token: string): string {
    if (token === '[CLS]')
      return 'Always first. It belongs to no word, so its final vector can summarise the whole input for a classifier.';
    if (token === '[SEP]') return 'Marks the end of a sentence.';
    if (token.startsWith('##')) return 'A word piece: "##" means it continues the token before it.';
    return 'An ordinary token: one row of the 30,000-row WordPiece embedding table.';
  }
}
