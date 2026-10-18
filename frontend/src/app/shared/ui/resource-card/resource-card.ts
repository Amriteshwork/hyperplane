import { ChangeDetectionStrategy, Component, computed, inject, input, signal } from '@angular/core';
import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser';
import { NgIcon, provideIcons } from '@ng-icons/core';
import {
  lucideExternalLink,
  lucideFileText,
  lucideGraduationCap,
  lucideMousePointerClick,
  lucidePlay,
  lucideScrollText,
  lucideVideo,
} from '@ng-icons/lucide';
import { HlmBadge } from '@spartan-ng/helm/badge';

import { youtubeId } from './youtube';

export type ResourceType = 'blog' | 'video' | 'paper' | 'interactive' | 'course';
export type ResourceLevel = 'beginner' | 'intermediate' | 'advanced';

export interface LearningResource {
  readonly type: ResourceType;
  readonly title: string;
  readonly url: string;
  readonly author?: string | null;
  readonly level: ResourceLevel;
  readonly durationMin?: number | null;
  /** One line on why it's worth the learner's time. */
  readonly why?: string | null;
}

const ICONS: Record<ResourceType, string> = {
  blog: 'lucideFileText',
  video: 'lucideVideo',
  paper: 'lucideScrollText',
  interactive: 'lucideMousePointerClick',
  course: 'lucideGraduationCap',
};

/** A curated link; YouTube videos get a click-to-load embed (no third-party request until then). */
@Component({
  selector: 'app-resource-card',
  imports: [NgIcon, HlmBadge],
  providers: [
    provideIcons({
      lucideExternalLink,
      lucideFileText,
      lucideGraduationCap,
      lucideMousePointerClick,
      lucidePlay,
      lucideScrollText,
      lucideVideo,
    }),
  ],
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: { class: 'block' },
  template: `
    <article class="flex h-full flex-col overflow-hidden rounded-lg border bg-card">
      @if (videoId(); as id) {
        <div class="aspect-video bg-muted">
          @if (embedUrl(); as src) {
            <iframe
              class="size-full"
              loading="lazy"
              [src]="src"
              [title]="resource().title"
              allow="accelerometer; encrypted-media; gyroscope; picture-in-picture"
              allowfullscreen
            ></iframe>
          } @else {
            <button
              type="button"
              class="group relative size-full focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring"
              (click)="playing.set(true)"
            >
              <img
                class="size-full object-cover"
                loading="lazy"
                alt=""
                [src]="'https://i.ytimg.com/vi/' + id + '/hqdefault.jpg'"
              />
              <span
                class="absolute inset-0 grid place-items-center bg-black/20 group-hover:bg-black/30"
              >
                <span
                  class="grid size-12 place-items-center rounded-full bg-background/90 text-foreground shadow"
                >
                  <ng-icon name="lucidePlay" size="1.25rem" aria-hidden="true" />
                </span>
              </span>
              <span class="sr-only">Play video: {{ resource().title }}</span>
            </button>
          }
        </div>
      }
      <div class="flex flex-1 flex-col gap-2 p-4">
        <div class="flex items-center gap-2 text-xs text-muted-foreground">
          <ng-icon [name]="icon()" aria-hidden="true" />
          <span class="capitalize">{{ resource().type }}</span>
          @if (resource().durationMin) {
            <span aria-hidden="true">·</span>
            <span>{{ resource().durationMin }} min</span>
          }
          <span hlmBadge variant="outline" class="ms-auto capitalize">{{ resource().level }}</span>
        </div>
        <h3 class="font-medium leading-snug">
          <a
            class="hover:underline focus-visible:outline-2 focus-visible:outline-ring"
            target="_blank"
            rel="noopener noreferrer"
            [href]="resource().url"
          >
            {{ resource().title }}
            <ng-icon
              name="lucideExternalLink"
              size="0.75rem"
              class="ms-1 align-baseline"
              aria-hidden="true"
            />
            <span class="sr-only">(opens in a new tab)</span>
          </a>
        </h3>
        @if (resource().author) {
          <p class="text-xs text-muted-foreground">{{ resource().author }}</p>
        }
        @if (resource().why) {
          <p class="text-sm text-muted-foreground">{{ resource().why }}</p>
        }
      </div>
    </article>
  `,
})
export class ResourceCard {
  private readonly sanitizer = inject(DomSanitizer);

  readonly resource = input.required<LearningResource>();

  protected readonly playing = signal(false);
  protected readonly icon = computed(() => ICONS[this.resource().type]);
  protected readonly videoId = computed(() =>
    this.resource().type === 'video' ? youtubeId(this.resource().url) : null,
  );
  protected readonly embedUrl = computed<SafeResourceUrl | null>(() => {
    const id = this.videoId();
    if (!id || !this.playing()) return null;
    // Safe: `id` is validated to 11 URL-safe characters, and the host is fixed.
    return this.sanitizer.bypassSecurityTrustResourceUrl(
      `https://www.youtube-nocookie.com/embed/${id}?autoplay=1`,
    );
  });
}
