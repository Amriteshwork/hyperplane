import { ChangeDetectionStrategy, Component, computed, input, output, signal } from '@angular/core';
import { NgIcon, provideIcons } from '@ng-icons/core';
import { lucideUpload } from '@ng-icons/lucide';
import { HlmProgressImports } from '@spartan-ng/helm/progress';

let nextDropzoneId = 0;

/** Drag-and-drop or browse for one file; checks extension and size before emitting it. */
@Component({
  selector: 'app-file-dropzone',
  imports: [NgIcon, HlmProgressImports],
  providers: [provideIcons({ lucideUpload })],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <label
      class="flex cursor-pointer flex-col items-center gap-2 rounded-lg border border-dashed px-4 py-8 text-center transition-colors has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-ring motion-reduce:transition-none"
      [class.bg-muted]="dragging()"
      [class.border-primary]="dragging()"
      [class.pointer-events-none]="busy()"
      [class.opacity-60]="busy()"
      [for]="inputId"
      (dragenter)="onDrag($event, true)"
      (dragover)="onDrag($event, true)"
      (dragleave)="onDrag($event, false)"
      (drop)="onDrop($event)"
    >
      <ng-icon name="lucideUpload" size="1.5rem" class="text-muted-foreground" aria-hidden="true" />
      <span class="text-sm font-medium"
        >Drop a file here or <span class="underline">browse</span></span
      >
      <span class="text-xs text-muted-foreground" [id]="inputId + '-hint'">{{ hint() }}</span>
      <input
        class="sr-only"
        type="file"
        [id]="inputId"
        [accept]="accept()"
        [disabled]="busy()"
        [attr.aria-describedby]="inputId + '-hint' + (error() ? ' ' + inputId + '-error' : '')"
        (change)="onPick($event)"
      />
    </label>
    @if (progress() !== null) {
      <div class="mt-3 grid gap-1">
        <hlm-progress [value]="progress()" aria-label="Upload progress">
          <hlm-progress-indicator />
        </hlm-progress>
        <span class="text-xs text-muted-foreground tabular-nums">Uploading… {{ progress() }}%</span>
      </div>
    }
    @if (error()) {
      <p class="mt-2 text-sm text-destructive" role="alert" [id]="inputId + '-error'">
        {{ error() }}
      </p>
    }
  `,
})
export class FileDropzone {
  /** Comma-separated extensions, e.g. `.csv,.parquet,.xlsx`. */
  readonly accept = input('');
  readonly maxBytes = input<number | null>(null);
  /** 0–100 while uploading; `null` when idle. */
  readonly progress = input<number | null>(null);
  /** An error from the caller (e.g. the server rejected the file). */
  readonly errorText = input<string | null>(null);
  readonly fileSelect = output<File>();

  protected readonly inputId = `dropzone-${nextDropzoneId++}`;
  protected readonly dragging = signal(false);
  private readonly localError = signal<string | null>(null);
  protected readonly error = computed(() => this.localError() ?? this.errorText());
  protected readonly busy = computed(() => this.progress() !== null);

  protected readonly hint = computed(() => {
    const parts = [];
    if (this.accept()) parts.push(this.accept().split(',').join(', '));
    const max = this.maxBytes();
    if (max) parts.push(`up to ${formatBytes(max)}`);
    return parts.join(' · ');
  });

  protected onDrag(event: DragEvent, over: boolean): void {
    event.preventDefault();
    this.dragging.set(over);
  }

  protected onDrop(event: DragEvent): void {
    event.preventDefault();
    this.dragging.set(false);
    const file = event.dataTransfer?.files.item(0);
    if (file) this.take(file);
  }

  protected onPick(event: Event): void {
    const el = event.target as HTMLInputElement;
    const file = el.files?.item(0);
    if (file) this.take(file);
    el.value = ''; // let the same file be picked again after an error
  }

  private take(file: File): void {
    const problem = this.check(file);
    this.localError.set(problem);
    if (!problem) this.fileSelect.emit(file);
  }

  private check(file: File): string | null {
    const allowed = this.accept()
      .split(',')
      .map((ext) => ext.trim().toLowerCase())
      .filter(Boolean);
    const name = file.name.toLowerCase();
    if (allowed.length && !allowed.some((ext) => name.endsWith(ext))) {
      return `“${file.name}” isn't a supported file type (${allowed.join(', ')}).`;
    }
    const max = this.maxBytes();
    if (max && file.size > max) return `“${file.name}” is larger than ${formatBytes(max)}.`;
    return null;
  }
}

function formatBytes(bytes: number): string {
  if (bytes >= 1024 ** 2) return `${Math.round(bytes / 1024 ** 2)} MB`;
  if (bytes >= 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${bytes} B`;
}
