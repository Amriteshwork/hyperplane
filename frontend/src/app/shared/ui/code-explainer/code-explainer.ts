import { NgTemplateOutlet } from '@angular/common';
import {
  ChangeDetectionStrategy,
  Component,
  DOCUMENT,
  computed,
  inject,
  input,
  linkedSignal,
  resource,
  signal,
} from '@angular/core';
import { RouterLink } from '@angular/router';
import { NgIcon, provideIcons } from '@ng-icons/core';
import {
  lucideCheck,
  lucideChevronLeft,
  lucideChevronRight,
  lucideCopy,
  lucideDownload,
  lucideX,
} from '@ng-icons/lucide';
import { HlmButton } from '@spartan-ng/helm/button';

import { saveFile } from '../../browser/save-file';
import { MarkdownView } from '../markdown-view/markdown-view';
import { CodeLanguage, CodeLine, highlightLines, plainLines } from './highlight';

export interface ConceptChip {
  readonly id: string;
  readonly title: string;
}

/** One explained span. Line numbers come from anchors resolved server-side, never authored. */
export interface CodeAnnotation {
  readonly id: string;
  /** 1-based, inclusive. */
  readonly startLine: number;
  readonly endLine: number;
  readonly short: string;
  /** Markdown. */
  readonly deep: string;
  readonly concepts: readonly ConceptChip[];
}

export type ExplanationDepth = 'short' | 'deep';

interface Segment {
  readonly start: number;
  readonly annotation: CodeAnnotation | null;
  readonly lines: readonly CodeLine[];
}

let nextExplainerId = 0;

/** Code on the left, the selected block's explanation on the right (stacked on mobile). */
@Component({
  selector: 'app-code-explainer',
  imports: [NgTemplateOutlet, RouterLink, NgIcon, HlmButton, MarkdownView],
  providers: [
    provideIcons({
      lucideCheck,
      lucideChevronLeft,
      lucideChevronRight,
      lucideCopy,
      lucideDownload,
      lucideX,
    }),
  ],
  changeDetection: ChangeDetectionStrategy.OnPush,
  styleUrl: './code-explainer.css',
  template: `
    <div class="grid gap-4 lg:items-start" [class]="columns()">
      <section
        class="min-w-0 overflow-hidden rounded-lg border bg-card"
        [attr.aria-label]="filename()"
      >
        <header class="flex items-center justify-between gap-2 border-b px-3 py-2">
          <span class="truncate font-mono text-xs text-muted-foreground">{{ filename() }}</span>
          <span class="flex shrink-0 gap-1">
            <button hlmBtn variant="ghost" size="sm" type="button" (click)="copy()">
              <ng-icon [name]="copied() ? 'lucideCheck' : 'lucideCopy'" aria-hidden="true" />
              <span>{{ copied() ? 'Copied' : 'Copy' }}</span>
            </button>
            <button hlmBtn variant="ghost" size="sm" type="button" (click)="download()">
              <ng-icon name="lucideDownload" aria-hidden="true" />
              <span>Download</span>
            </button>
          </span>
        </header>
        <div class="overflow-x-auto py-2">
          <code class="code block w-max min-w-full text-[13px] leading-6">
            @for (segment of segments(); track segment.start) {
              @if (segment.annotation; as annotation) {
                <button
                  type="button"
                  class="segment block w-full text-left"
                  [attr.data-anchor]="annotation.id"
                  [attr.aria-pressed]="annotation.id === selectedId()"
                  [attr.aria-controls]="panelId"
                  (click)="select(annotation.id)"
                >
                  <ng-container *ngTemplateOutlet="lineList; context: { $implicit: segment }" />
                </button>
              } @else {
                <ng-container *ngTemplateOutlet="lineList; context: { $implicit: segment }" />
              }
            }
          </code>
        </div>
      </section>

      @if (annotations().length) {
        <aside
          class="panel rounded-lg border bg-card p-4 max-lg:sticky max-lg:bottom-2 max-lg:max-h-[45vh] max-lg:overflow-y-auto max-lg:shadow-lg lg:sticky lg:top-20"
          aria-live="polite"
          [id]="panelId"
        >
          @if (selected(); as annotation) {
            <div class="flex flex-wrap items-center justify-between gap-2">
              <p class="flex items-center gap-1 font-mono text-xs text-muted-foreground">
                <button
                  hlmBtn
                  variant="ghost"
                  size="icon-sm"
                  type="button"
                  aria-label="Close explanation"
                  (click)="selectedId.set(null)"
                >
                  <ng-icon name="lucideX" aria-hidden="true" />
                </button>
                {{ lineRange(annotation) }}
              </p>
              <div
                class="inline-flex rounded-md border p-0.5"
                role="group"
                aria-label="Explanation depth"
              >
                @for (option of depths; track option) {
                  <button
                    type="button"
                    class="rounded px-2.5 py-1 text-xs font-medium capitalize text-muted-foreground aria-pressed:bg-muted aria-pressed:text-foreground"
                    [attr.aria-pressed]="depth() === option"
                    (click)="depth.set(option)"
                  >
                    {{ option }}
                  </button>
                }
              </div>
            </div>
            <app-markdown-view
              class="mt-3 block"
              [content]="depth() === 'short' ? annotation.short : annotation.deep"
            />
            @if (annotation.concepts.length) {
              <ul class="mt-4 flex flex-wrap gap-2" aria-label="Related concepts">
                @for (concept of annotation.concepts; track concept.id) {
                  <li>
                    <a
                      class="inline-flex rounded-full border px-2.5 py-0.5 text-xs hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring"
                      [routerLink]="[conceptBasePath(), concept.id]"
                    >
                      {{ concept.title }}
                    </a>
                  </li>
                }
              </ul>
            }
            <div class="mt-4 flex justify-between gap-2 border-t pt-3">
              <button
                hlmBtn
                variant="ghost"
                size="sm"
                type="button"
                [disabled]="!previous()"
                (click)="step(-1)"
              >
                <ng-icon name="lucideChevronLeft" aria-hidden="true" /> Previous
              </button>
              <button
                hlmBtn
                variant="ghost"
                size="sm"
                type="button"
                [disabled]="!next()"
                (click)="step(1)"
              >
                Next <ng-icon name="lucideChevronRight" aria-hidden="true" />
              </button>
            </div>
          } @else {
            <p class="text-sm text-muted-foreground">
              Select a highlighted block of code to see what it does and why.
            </p>
            <button hlmBtn variant="outline" size="sm" type="button" class="mt-3" (click)="step(1)">
              Start from the top
            </button>
          }
        </aside>
      }
    </div>

    <ng-template #lineList let-segment>
      @for (line of segment.lines; track $index) {
        <span class="line">
          <span class="line-number" aria-hidden="true">{{ segment.start + $index }}</span>
          @for (token of line; track $index) {
            <span class="token" [style]="token.style">{{ token.content }}</span>
          }
        </span>
      }
    </ng-template>
  `,
})
export class CodeExplainer {
  private readonly document = inject(DOCUMENT);

  readonly code = input.required<string>();
  readonly annotations = input<readonly CodeAnnotation[]>([]);
  readonly filename = input('train.py');
  readonly language = input<CodeLanguage>('python');
  readonly conceptBasePath = input('/concepts');
  readonly initialDepth = input<ExplanationDepth>('short');

  protected readonly panelId = `code-explainer-${nextExplainerId++}`;
  protected readonly depths: readonly ExplanationDepth[] = ['short', 'deep'];
  protected readonly depth = linkedSignal(() => this.initialDepth());
  protected readonly copied = signal(false);
  /** Resets when the code changes, e.g. switching train → scratch. */
  protected readonly selectedId = linkedSignal<string, string | null>({
    source: () => this.code(),
    computation: () => null,
  });

  private readonly highlighted = resource({
    params: () => ({ code: this.code(), lang: this.language() }),
    loader: ({ params }) => highlightLines(params.code, params.lang),
  });

  /** Plain code (no annotations) gets the full width: there's nothing to explain beside it. */
  protected readonly columns = computed(() =>
    this.annotations().length ? 'lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]' : '',
  );
  protected readonly lines = computed<readonly CodeLine[]>(() =>
    this.highlighted.hasValue() ? this.highlighted.value() : plainLines(this.code()),
  );

  protected readonly ordered = computed(() =>
    [...this.annotations()].sort((a, b) => a.startLine - b.startLine),
  );

  protected readonly segments = computed(() => toSegments(this.lines(), this.ordered()));

  protected readonly selected = computed(
    () => this.ordered().find((a) => a.id === this.selectedId()) ?? null,
  );
  private readonly selectedIndex = computed(() => {
    const id = this.selectedId();
    return this.ordered().findIndex((a) => a.id === id);
  });
  protected readonly previous = computed(() => this.ordered()[this.selectedIndex() - 1] ?? null);
  protected readonly next = computed(() => this.ordered()[this.selectedIndex() + 1] ?? null);

  protected select(id: string): void {
    this.selectedId.set(id);
  }

  protected step(direction: 1 | -1): void {
    const target = direction === 1 ? this.next() : this.previous();
    if (target) this.selectedId.set(target.id);
  }

  protected lineRange(annotation: CodeAnnotation): string {
    return annotation.startLine === annotation.endLine
      ? `Line ${annotation.startLine}`
      : `Lines ${annotation.startLine}–${annotation.endLine}`;
  }

  protected async copy(): Promise<void> {
    await this.document.defaultView?.navigator.clipboard?.writeText(this.code());
    this.copied.set(true);
    setTimeout(() => this.copied.set(false), 2000);
  }

  protected download(): void {
    saveFile(this.document, new Blob([this.code()], { type: 'text/x-python' }), this.filename());
  }
}

/** Groups lines into annotated blocks and plain runs; overlapping spans keep the first claim. */
function toSegments(lines: readonly CodeLine[], annotations: readonly CodeAnnotation[]): Segment[] {
  const segments: Segment[] = [];
  let cursor = 1;
  const pushPlain = (until: number) => {
    if (until > cursor)
      segments.push({ start: cursor, annotation: null, lines: lines.slice(cursor - 1, until - 1) });
  };
  for (const annotation of annotations) {
    const start = Math.max(annotation.startLine, cursor);
    const end = Math.min(annotation.endLine, lines.length);
    if (start > end) continue;
    pushPlain(start);
    segments.push({ start, annotation, lines: lines.slice(start - 1, end) });
    cursor = end + 1;
  }
  pushPlain(lines.length + 1);
  return segments;
}
