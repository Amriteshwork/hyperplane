import { Injectable, computed, inject, resource, signal } from '@angular/core';

import { Api } from '../../core/api/api';
import { getTrack } from '../../core/api/fn/learn/get-track';
import { errorMessage, isNotFound } from '../../core/http/http-errors';
import { splitPapers, toLearningResource } from '../../shared/learn/mappers';

@Injectable()
export class TrackStore {
  private readonly api = inject(Api);

  readonly trackId = signal<string | null>(null);

  private readonly detail = resource({
    params: () => this.trackId() ?? undefined,
    loader: ({ params }) => this.api.invoke(getTrack, { track_id: params }),
  });

  readonly track = computed(() => (this.detail.hasValue() ? this.detail.value() : null));
  /** Chapters with the step number of their first concept: numbering runs across the track. */
  readonly chapters = computed(() => {
    let start = 1;
    return (this.track()?.chapters ?? []).map((chapter) => {
      const numbered = { ...chapter, start };
      start += chapter.concepts.length;
      return numbered;
    });
  });
  readonly conceptCount = computed(() =>
    this.chapters().reduce((n, c) => n + c.concepts.length, 0),
  );
  readonly firstConcept = computed(() => this.chapters()[0]?.concepts[0] ?? null);
  private readonly split = computed(() =>
    splitPapers(this.track()?.resources.map(toLearningResource) ?? []),
  );
  readonly resources = computed(() => this.split().others);
  readonly papers = computed(() => this.split().papers);
  readonly notFound = computed(() => isNotFound(this.detail.error()));
  readonly error = computed(() => (this.detail.error() ? errorMessage(this.detail.error()) : null));

  reload(): void {
    this.detail.reload();
  }
}
