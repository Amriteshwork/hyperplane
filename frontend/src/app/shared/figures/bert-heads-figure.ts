import { DecimalPipe } from '@angular/common';
import { ChangeDetectionStrategy, Component, computed, signal } from '@angular/core';

import { FigureFrame } from './figure-frame';

/** BERT-base's width and parameter count. */
const H = 768;
const PRETRAINED = 110_000_000;

interface Column {
  token: string;
  segment: 0 | 1;
  /** What the head outputs at this position, if it reads it. */
  out?: string;
}

interface Task {
  label: string;
  question: string;
  columns: Column[];
  head: string;
  /** Size of the new weight matrix: rows × H. */
  rows: number;
  why: string;
}

function columns(spec: string, outs: Record<number, string> = {}): Column[] {
  let segment: 0 | 1 = 0;
  return spec.split(' ').map((token, i) => {
    const column: Column = { token, segment, out: outs[i] };
    if (token === '[SEP]') segment = 1;
    return column;
  });
}

const TASKS: readonly Task[] = [
  {
    label: 'Sentiment',
    question: 'Is this review positive?',
    columns: columns('[CLS] the film was great [SEP]', { 0: 'positive 97%' }),
    head: 'softmax(C·Wᵀ) on the [CLS] vector C, with K = 2 classes',
    rows: 2,
    why: 'One label for the whole input, so only the [CLS] vector is read. Self-attention has already gathered the sentence into it.',
  },
  {
    label: 'Entailment',
    question: 'Does B follow from A?',
    columns: columns('[CLS] a man sleeps [SEP] someone rests [SEP]', { 0: 'follows 91%' }),
    head: 'softmax(C·Wᵀ) on [CLS], K = 3: follows, contradicts, neutral',
    rows: 3,
    why: 'Two sentences, packed as A and B. Every word of A attends to every word of B, so comparing them needs no extra machinery.',
  },
  {
    label: 'Names & places',
    question: 'Tag every word',
    columns: columns('[CLS] raj flew to paris [SEP]', {
      1: 'PERSON',
      2: 'other',
      3: 'other',
      4: 'PLACE',
    }),
    head: 'softmax(Tᵢ·Wᵀ) at every token i, one tag each',
    rows: 9,
    why: 'One label per token, so the same small classifier runs on every token’s vector. Each vector already knows its context: "paris" after "flew to" is a place.',
  },
  {
    label: 'Question answering',
    question: 'Where is the answer?',
    columns: columns('[CLS] where is the cat [SEP] on the mat [SEP]', {
      6: 'start',
      8: 'end',
    }),
    head: 'start = argmax S·Tᵢ, end = argmax E·Tⱼ (j ≥ i), over the passage',
    rows: 2,
    why: 'Question as A, passage as B. Two learned vectors, S and E, score every passage token as the start or end of the answer. The answer is copied out of the passage, never generated.',
  },
];

/** One pretrained BERT, four tasks: which output vectors each task's small head reads. */
@Component({
  selector: 'app-bert-heads-figure',
  imports: [DecimalPipe, FigureFrame],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <app-figure-frame
      heading="One model, many heads"
      hint="Pick a task. Coloured vectors are the ones its head reads."
    >
      <div class="flex flex-wrap gap-1.5" role="group" aria-label="Fine-tuning task">
        @for (t of tasks; track t.label; let i = $index) {
          <button
            type="button"
            class="rounded-md border px-2.5 py-1 text-xs hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring"
            [class.bg-muted]="selected() === i"
            [attr.aria-pressed]="selected() === i"
            (click)="selected.set(i)"
          >
            {{ t.label }}
          </button>
        }
      </div>

      <p class="mt-3 text-sm font-medium">{{ task().question }}</p>
      <div class="mt-2 overflow-x-auto pb-1" role="img" [attr.aria-label]="description()">
        <div class="inline-grid min-w-full gap-y-1.5" [style.grid-template-columns]="grid()">
          @for (c of task().columns; track $index) {
            <span
              class="mx-0.5 self-end rounded px-1 py-0.5 text-center text-[10px] leading-tight font-semibold break-words"
              [class]="c.out ? 'bg-fig-2/20 text-foreground' : 'text-transparent'"
            >
              {{ c.out ?? '·' }}
            </span>
          }
          @for (c of task().columns; track $index) {
            <span
              class="mx-auto block size-4 rounded-sm"
              [class]="c.out ? 'bg-fig-2' : 'bg-muted-foreground/25'"
            ></span>
          }
          <span
            class="rounded-md border border-fig-1/60 bg-fig-1/10 py-1.5 text-center text-xs font-semibold text-fig-1"
            [style.grid-column]="'1 / -1'"
          >
            pretrained BERT (shared by every task)
          </span>
          @for (c of task().columns; track $index) {
            <span
              class="mx-0.5 rounded border px-1 py-0.5 text-center font-mono text-[10px]"
              [class]="c.segment === 0 ? 'border-fig-3/60' : 'border-fig-4/60'"
            >
              {{ c.token }}
            </span>
          }
        </div>
      </div>

      <div class="mt-3 rounded-md bg-muted/50 p-3 text-sm" aria-live="polite">
        <p><span class="font-semibold">Head:</span> {{ task().head }}</p>
        <p class="mt-1 text-xs text-muted-foreground">{{ task().why }}</p>
        <p class="mt-1 text-xs text-muted-foreground">
          New weights at BERT-base size: {{ task().rows }} × {{ h }} =
          <span class="text-foreground">{{ task().rows * h | number }}</span
          >, next to {{ pretrained | number }} pretrained ones.
        </p>
      </div>
      <p caption>
        The model in the middle is identical for every task: the same pretrained weights, copied,
        then fine-tuned for an hour or so on that task's labelled examples. Only the small head on
        top is new. Classification reads the [CLS] vector, tagging reads every token, and question
        answering picks a start and an end token in the passage. Token borders show the segment:
        green for sentence A, yellow for B.
      </p>
    </app-figure-frame>
  `,
})
export class BertHeadsFigure {
  protected readonly tasks = TASKS;
  protected readonly h = H;
  protected readonly pretrained = PRETRAINED;
  protected readonly selected = signal(0);
  protected readonly task = computed(() => TASKS[this.selected()]);
  protected readonly grid = computed(
    () => `repeat(${this.task().columns.length}, minmax(3.25rem, 1fr))`,
  );
  protected readonly description = computed(() => {
    const t = this.task();
    const read = t.columns.filter((c) => c.out).map((c) => `${c.token}: ${c.out}`);
    return `${t.label}: input ${t.columns.map((c) => c.token).join(' ')}. The head reads ${read.join(', ')}.`;
  });
}
