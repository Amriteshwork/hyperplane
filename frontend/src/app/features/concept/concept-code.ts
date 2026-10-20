import { ChangeDetectionStrategy, Component, computed, input, linkedSignal } from '@angular/core';

import { CodeLibrary } from '../../core/api/models/code-library';
import { ConceptCodeOut } from '../../core/api/models/concept-code-out';
import { toCodeAnnotations } from '../../shared/learn/mappers';
import { CodeExplainer } from '../../shared/ui/code-explainer/code-explainer';

/** A concept implemented from scratch on random data, one tab per library. */
@Component({
  selector: 'app-concept-code',
  imports: [CodeExplainer],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="inline-flex rounded-md border p-0.5" role="tablist" aria-label="Library">
      @for (implementation of code(); track implementation.library) {
        <button
          type="button"
          role="tab"
          class="rounded px-3 py-1 text-sm font-medium text-muted-foreground aria-selected:bg-muted aria-selected:text-foreground focus-visible:outline-2 focus-visible:outline-ring"
          [id]="'concept-code-tab-' + implementation.library"
          [attr.aria-selected]="implementation.library === selected()?.library"
          aria-controls="concept-code-panel"
          [tabIndex]="implementation.library === selected()?.library ? 0 : -1"
          (click)="library.set(implementation.library)"
          (keydown)="onKeydown($event)"
        >
          {{ implementation.title }}
        </button>
      }
    </div>
    @if (selected(); as implementation) {
      <div
        id="concept-code-panel"
        class="mt-3"
        role="tabpanel"
        [attr.aria-labelledby]="'concept-code-tab-' + implementation.library"
      >
        <app-code-explainer
          [code]="implementation.code"
          [annotations]="annotations()"
          [filename]="implementation.filename"
        />
      </div>
    }
  `,
})
export class ConceptCode {
  readonly code = input.required<readonly ConceptCodeOut[]>();

  /** Back to the first library whenever another concept's code arrives. */
  protected readonly library = linkedSignal<CodeLibrary | null>(
    () => this.code()[0]?.library ?? null,
  );
  protected readonly selected = computed(
    () => this.code().find((c) => c.library === this.library()) ?? null,
  );
  protected readonly annotations = computed(() => {
    const selected = this.selected();
    return selected ? toCodeAnnotations(selected) : [];
  });

  /** Arrow keys move between tabs, as the ARIA tabs pattern expects. */
  protected onKeydown(event: KeyboardEvent): void {
    const step = { ArrowRight: 1, ArrowLeft: -1 }[event.key];
    if (!step) return;
    event.preventDefault();
    const code = this.code();
    const index = code.findIndex((c) => c.library === this.library());
    const next = code[(index + step + code.length) % code.length];
    this.library.set(next.library);
    (event.currentTarget as HTMLElement).parentElement
      ?.querySelector<HTMLElement>(`#concept-code-tab-${next.library}`)
      ?.focus();
  }
}
