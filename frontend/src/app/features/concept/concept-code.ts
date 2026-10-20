import {
  ChangeDetectionStrategy,
  Component,
  DOCUMENT,
  computed,
  inject,
  input,
  linkedSignal,
  signal,
} from '@angular/core';
import { RouterLink } from '@angular/router';
import { NgIcon, provideIcons } from '@ng-icons/core';
import { lucideDownload } from '@ng-icons/lucide';
import { HlmButton } from '@spartan-ng/helm/button';

import { Api } from '../../core/api/api';
import { getConceptCodeBundle } from '../../core/api/fn/learn/get-concept-code-bundle';

import { CodeLibrary } from '../../core/api/models/code-library';
import { ConceptCodeOut } from '../../core/api/models/concept-code-out';
import { errorMessage } from '../../core/http/http-errors';
import { saveFile } from '../../shared/browser/save-file';
import { toCodeAnnotations } from '../../shared/learn/mappers';
import { CodeExplainer } from '../../shared/ui/code-explainer/code-explainer';

/** A concept implemented from scratch on random data, one tab per library. */
@Component({
  selector: 'app-concept-code',
  imports: [RouterLink, NgIcon, HlmButton, CodeExplainer],
  providers: [provideIcons({ lucideDownload })],
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
        @if (implementation.uses.length) {
          <div class="mb-3 rounded-md border bg-muted/40 px-3 py-2 text-sm" data-testid="code-uses">
            <p class="text-muted-foreground">
              Builds on earlier concepts. To run it, save these files next to this one, or download
              them all at once:
            </p>
            <ul class="mt-1 grid gap-1">
              @for (use of implementation.uses; track use.concept.id) {
                <li>
                  @for (name of use.names; track name; let last = $last) {
                    <code class="text-xs">{{ name }}</code
                    >{{ last ? '' : ', ' }}
                  }
                  from
                  <a
                    class="underline hover:text-foreground focus-visible:outline-2 focus-visible:outline-ring"
                    [routerLink]="['/concepts', use.concept.id]"
                    >{{ use.concept.title }}</a
                  >
                  <span class="ms-1 text-xs text-muted-foreground">({{ use.filename }})</span>
                </li>
              }
            </ul>
            <div class="mt-2 flex flex-wrap items-center gap-3">
              <button
                hlmBtn
                variant="outline"
                size="sm"
                type="button"
                [disabled]="bundling()"
                (click)="downloadBundle(implementation)"
              >
                <ng-icon name="lucideDownload" aria-hidden="true" />
                <span>{{ bundling() ? 'Preparing…' : 'Download with dependencies (.zip)' }}</span>
              </button>
              @if (bundleError(); as error) {
                <span class="text-xs text-destructive" role="alert">{{ error }}</span>
              }
            </div>
          </div>
        }
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
  private readonly api = inject(Api);
  private readonly document = inject(DOCUMENT);

  readonly conceptId = input.required<string>();
  readonly code = input.required<readonly ConceptCodeOut[]>();

  protected readonly bundling = signal(false);
  protected readonly bundleError = signal<string | null>(null);

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

  /** One zip: this file, every file it imports, and a README saying how to run it. */
  protected async downloadBundle(implementation: ConceptCodeOut): Promise<void> {
    const { library, filename } = implementation;
    this.bundling.set(true);
    this.bundleError.set(null);
    try {
      const zip = await this.api.invoke(getConceptCodeBundle, {
        concept_id: this.conceptId(),
        library,
      });
      saveFile(this.document, zip, filename.replace(/\.py$/, '.zip'));
    } catch (error) {
      this.bundleError.set(errorMessage(error));
    } finally {
      this.bundling.set(false);
    }
  }

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
