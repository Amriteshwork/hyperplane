import { ChangeDetectionStrategy, Component, computed, inject, input, signal } from '@angular/core';
import { DomSanitizer } from '@angular/platform-browser';

export interface VideoClipData {
  title: string;
  /** Null until the render is uploaded. */
  youtubeId: string | null;
}

const YOUTUBE_ID = /^[A-Za-z0-9_-]{11}$/;

/**
 * A short Manim animation hosted on YouTube (ADR 0022). Nothing loads from YouTube until the
 * reader presses play, so pages stay fast and cookie-free; videos not yet uploaded say so.
 * No host classes: it mounts into a `.video-slot` (styled block) and must keep that class.
 */
@Component({
  selector: 'app-video-clip',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <figure class="overflow-hidden rounded-lg border bg-card">
      <div class="relative aspect-video bg-muted">
        @if (src(); as src) {
          <iframe
            class="absolute inset-0 size-full"
            [src]="src"
            [title]="video().title"
            allow="autoplay; encrypted-media; picture-in-picture"
            allowfullscreen
            referrerpolicy="strict-origin-when-cross-origin"
          ></iframe>
        } @else if (playable()) {
          <button
            type="button"
            class="group absolute inset-0 flex flex-col items-center justify-center gap-3 focus-visible:outline-2 focus-visible:-outline-offset-4 focus-visible:outline-ring"
            (click)="playing.set(true)"
          >
            <span
              class="flex size-14 items-center justify-center rounded-full bg-primary text-primary-foreground transition-transform group-hover:scale-105 motion-reduce:transition-none"
              aria-hidden="true"
            >
              <svg viewBox="0 0 24 24" class="ml-1 size-6 fill-current">
                <path d="M8 5v14l11-7z" />
              </svg>
            </span>
            <span class="text-sm font-medium">Play animation</span>
            <span class="sr-only">: {{ video().title }} (loads YouTube)</span>
          </button>
        } @else {
          <p
            class="absolute inset-0 flex items-center justify-center p-4 text-center text-sm text-muted-foreground"
          >
            Animation coming soon.
          </p>
        }
      </div>
      <figcaption class="border-t px-4 py-2 text-sm text-muted-foreground">
        {{ video().title }}
      </figcaption>
    </figure>
  `,
})
export class VideoClip {
  readonly video = input.required<VideoClipData>();

  private readonly sanitizer = inject(DomSanitizer);
  protected readonly playing = signal(false);
  protected readonly playable = computed(() => YOUTUBE_ID.test(this.video().youtubeId ?? ''));
  /** Only a validated id reaches the trusted URL. */
  protected readonly src = computed(() =>
    this.playing() && this.playable()
      ? this.sanitizer.bypassSecurityTrustResourceUrl(
          `https://www.youtube-nocookie.com/embed/${this.video().youtubeId}?autoplay=1&rel=0`,
        )
      : null,
  );
}
