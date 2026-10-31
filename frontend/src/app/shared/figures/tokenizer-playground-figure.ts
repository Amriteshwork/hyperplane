import { ChangeDetectionStrategy, Component, computed, signal } from '@angular/core';
import { Tiktoken, TiktokenBPE } from 'js-tiktoken/lite';

import { FigureFrame } from './figure-frame';

interface EncodingOption {
  id: string;
  label: string;
  size: string;
  load: () => Promise<{ default: TiktokenBPE }>;
}

/** Vocabularies are large, so each loads only when a reader picks it. */
const ENCODINGS: readonly EncodingOption[] = [
  { id: 'gpt2', label: 'GPT-2', size: '50,257', load: () => import('js-tiktoken/ranks/gpt2') },
  {
    id: 'cl100k_base',
    label: 'GPT-4',
    size: 'about 100,000',
    load: () => import('js-tiktoken/ranks/cl100k_base'),
  },
  {
    id: 'o200k_base',
    label: 'GPT-4o',
    size: 'about 200,000',
    load: () => import('js-tiktoken/ranks/o200k_base'),
  },
];

const PRESETS: readonly { label: string; text: string }[] = [
  { label: 'A sentence', text: 'The cat sat on the mat.' },
  { label: 'Strawberry', text: 'How many r are in strawberry?' },
  { label: 'Numbers', text: '1234 + 1235 = 2469' },
  { label: 'Rare words', text: 'tokenization is unbelievably antidisestablishmentarian' },
  { label: 'Capitals', text: 'hello Hello HELLO  hello' },
  { label: 'Code', text: 'for i in range(10):\n    print(i)' },
  { label: 'Emoji', text: 'I love 🍕 and 🦙!' },
  { label: 'Hindi', text: 'नमस्ते दुनिया' },
];

export interface TokenChip {
  id: number;
  text: string;
}

/** One chip per token. A token holding part of a character decodes to U+FFFD, shown as ◆. */
export function tokenChips(encoding: Tiktoken, text: string): TokenChip[] {
  return encoding.encode(text).map((id) => ({ id, text: encoding.decode([id]) }));
}

/** Type anything and see how OpenAI's real tokenizers cut it up, in the reader's own browser. */
@Component({
  selector: 'app-tokenizer-playground-figure',
  imports: [FigureFrame],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <app-figure-frame
      heading="A real tokenizer (tiktoken)"
      hint="Type anything, or pick an example. Nothing leaves your browser."
    >
      <div class="flex flex-wrap gap-1.5" role="group" aria-label="Examples">
        @for (preset of presets; track preset.label) {
          <button
            type="button"
            class="rounded-md border px-2 py-1 text-xs hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring"
            (click)="text.set(preset.text)"
          >
            {{ preset.label }}
          </button>
        }
      </div>
      <label class="mt-2 block text-xs">
        <span class="sr-only">Text to tokenize</span>
        <textarea
          rows="2"
          class="block w-full rounded-md border bg-background p-2 font-mono text-sm focus-visible:outline-2 focus-visible:outline-ring"
          [value]="text()"
          (input)="onInput($event)"
        ></textarea>
      </label>

      <div class="mt-2 flex flex-wrap items-center gap-1.5" role="group" aria-label="Tokenizer">
        @for (option of encodings; track option.id) {
          <button
            type="button"
            class="rounded-md border px-3 py-1.5 text-xs hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring"
            [class.bg-muted]="selected() === option.id"
            [attr.aria-pressed]="selected() === option.id"
            (click)="select(option)"
          >
            {{ option.label }}
          </button>
        }
        <button
          type="button"
          class="rounded-md border px-3 py-1.5 text-xs hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring"
          [attr.aria-pressed]="showIds()"
          (click)="showIds.set(!showIds())"
        >
          {{ showIds() ? 'Show text' : 'Show ids' }}
        </button>
      </div>

      <div class="mt-3 min-h-16 rounded-md bg-muted/50 p-2" aria-live="polite">
        @if (encoding()) {
          <div class="flex flex-wrap gap-0.5 font-mono text-sm" data-testid="tokens">
            @for (chip of chips(); track $index) {
              <span class="rounded px-1 py-0.5 whitespace-pre-wrap" [class]="tint($index)">{{
                showIds() ? chip.id : visible(chip.text)
              }}</span>
            }
          </div>
          <p class="mt-2 text-xs text-muted-foreground">
            <span class="font-semibold text-foreground">{{ chips().length }}</span>
            tokens · {{ text().length }} characters · a vocabulary of
            {{ vocabulary() }}
          </p>
        } @else if (failed()) {
          <p class="text-sm">Couldn't load this tokenizer's vocabulary. Try another.</p>
        } @else {
          <p class="text-sm text-muted-foreground">Loading the vocabulary…</p>
        }
      </div>
      <p caption>
        Each colour is one token, the only unit the model ever sees. Try "Strawberry": the whole
        word is one token, so the model never looks at its letters, which is why counting r's is
        hard for it. Try "Numbers": GPT-2 splits 1234 into 12 and 34. Try "Capitals": hello and
        HELLO are unrelated tokens. "Emoji" and "Hindi" can show ◆, where a token holds part of a
        character, because tokenizers work on bytes. Then compare tokenizers on "Hindi": GPT-2 needs
        23 tokens and GPT-4o needs 5, because its larger vocabulary was built from more languages.
      </p>
    </app-figure-frame>
  `,
})
export class TokenizerPlaygroundFigure {
  protected readonly encodings = ENCODINGS;
  protected readonly presets = PRESETS;
  protected readonly text = signal(PRESETS[0].text);
  protected readonly selected = signal('gpt2');
  protected readonly showIds = signal(false);
  protected readonly encoding = signal<Tiktoken | null>(null);
  protected readonly failed = signal(false);
  protected readonly vocabulary = signal('');
  private request = 0;

  protected readonly chips = computed(() => {
    const encoding = this.encoding();
    return encoding ? tokenChips(encoding, this.text()) : [];
  });

  constructor() {
    void this.select(ENCODINGS[0]);
  }

  protected async select(option: EncodingOption): Promise<void> {
    const request = ++this.request;
    this.selected.set(option.id);
    this.encoding.set(null);
    this.failed.set(false);
    try {
      const ranks = (await option.load()).default;
      if (request !== this.request) return;
      this.encoding.set(new Tiktoken(ranks));
      this.vocabulary.set(option.size);
    } catch {
      if (request === this.request) this.failed.set(true);
    }
  }

  protected onInput(event: Event): void {
    this.text.set((event.target as HTMLTextAreaElement).value);
  }

  /** A newline or a space alone would be an invisible chip. */
  protected visible(text: string): string {
    return text.replaceAll('\n', '↵').replaceAll('�', '◆').replace(/^ $/, '␣');
  }

  protected tint(i: number): string {
    return ['bg-fig-1/25', 'bg-fig-2/25', 'bg-fig-3/25', 'bg-fig-4/25'][i % 4];
  }
}
