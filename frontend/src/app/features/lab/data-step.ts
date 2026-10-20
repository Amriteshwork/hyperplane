import { ChangeDetectionStrategy, Component, inject, output } from '@angular/core';
import { HlmBadge } from '@spartan-ng/helm/badge';
import { HlmButton } from '@spartan-ng/helm/button';
import { HlmProgressImports } from '@spartan-ng/helm/progress';

import { FileDropzone } from '../../shared/ui/file-dropzone/file-dropzone';
import { LabStore } from './lab.store';

const ACCEPTED_FILES = '.csv,.parquet,.xlsx';

/** Pick a sample or upload a file, then wait for profiling. */
@Component({
  selector: 'app-data-step',
  imports: [HlmBadge, HlmButton, HlmProgressImports, FileDropzone],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    @if (store.dataset(); as dataset) {
      @if (dataset.status === 'failed') {
        <div class="rounded-lg border border-destructive/50 p-4" role="alert">
          <p class="font-medium">Couldn’t read {{ dataset.name }}</p>
          <p class="mt-1 text-sm text-muted-foreground">{{ dataset.error }}</p>
          <button
            hlmBtn
            variant="outline"
            class="mt-3"
            type="button"
            (click)="chooseAnother.emit()"
          >
            Choose another dataset
          </button>
        </div>
      } @else {
        <div class="grid gap-3" role="status" aria-live="polite">
          <p class="text-sm">
            Profiling <span class="font-medium">{{ dataset.name }}</span
            >: column types, missing values and candidate targets…
          </p>
          <hlm-progress aria-label="Profiling"><hlm-progress-indicator /></hlm-progress>
        </div>
      }
    } @else {
      <section aria-labelledby="samples-heading">
        <h2 id="samples-heading" class="text-lg font-semibold">Start from a sample</h2>
        @if (store.datasetError(); as error) {
          <p class="mt-2 text-sm text-destructive" role="alert">{{ error }}</p>
        }
        @if (store.samplesError(); as error) {
          <p class="mt-2 text-sm text-destructive" role="alert">{{ error }}</p>
        }
        <ul class="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          @for (sample of store.samples(); track sample.id) {
            <li>
              <button
                type="button"
                class="flex h-full w-full flex-col items-start gap-2 rounded-lg border bg-card p-4 text-start hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring disabled:cursor-not-allowed disabled:opacity-50"
                [disabled]="sample.status !== 'available' || store.creatingDataset()"
                (click)="sampleSelect.emit(sample.id)"
              >
                <span class="flex w-full items-center gap-2">
                  <span class="font-medium">{{ sample.name }}</span>
                  <span hlmBadge variant="outline" class="ms-auto capitalize">{{
                    sample.task
                  }}</span>
                </span>
                <span class="text-sm text-muted-foreground">{{ sample.description }}</span>
                <span class="text-xs text-muted-foreground">
                  {{ sample.rows }} rows · predicts {{ sample.target }}
                  @if (sample.status !== 'available') {
                    · not available on this server
                  }
                </span>
              </button>
            </li>
          }
        </ul>
      </section>

      <section class="mt-10" aria-labelledby="upload-heading">
        <h2 id="upload-heading" class="text-lg font-semibold">Or upload your own</h2>
        <p class="mt-1 text-sm text-muted-foreground">
          CSV, Parquet or Excel; one row per example.
        </p>
        <app-file-dropzone
          class="mt-3 block"
          [accept]="acceptedFiles"
          [errorText]="store.createDatasetError()"
          (fileSelect)="fileSelect.emit($event)"
        />
        @if (store.creatingDataset()) {
          <p class="mt-2 text-sm text-muted-foreground" role="status">Uploading…</p>
        }
      </section>
    }
  `,
})
export class DataStep {
  protected readonly store = inject(LabStore);
  protected readonly acceptedFiles = ACCEPTED_FILES;

  readonly sampleSelect = output<string>();
  readonly fileSelect = output<File>();
  readonly chooseAnother = output();
}
