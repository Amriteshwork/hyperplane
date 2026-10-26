import { Injectable, computed, inject, resource, signal } from '@angular/core';

import { Api } from '../../core/api/api';
import { getPaper } from '../../core/api/fn/learn/get-paper';
import { errorMessage, isNotFound } from '../../core/http/http-errors';
import { toLearningResource } from '../../shared/learn/mappers';
import { VideoClipData } from '../../shared/ui/video-clip/video-clip';

@Injectable()
export class PaperStore {
  private readonly api = inject(Api);

  readonly paperId = signal<string | null>(null);

  private readonly detail = resource({
    params: () => this.paperId() ?? undefined,
    loader: ({ params }) => this.api.invoke(getPaper, { paper_id: params }),
  });

  readonly paper = computed(() => (this.detail.hasValue() ? this.detail.value() : null));
  readonly sections = computed(() => this.paper()?.sections ?? []);
  /** `::video[name]` lookups for every section's MarkdownView. */
  readonly videos = computed(() =>
    Object.fromEntries(
      (this.paper()?.videos ?? []).map((v): [string, VideoClipData] => [
        v.id,
        { title: v.title, youtubeId: v.youtube_id },
      ]),
    ),
  );
  readonly resources = computed(() => this.paper()?.resources.map(toLearningResource) ?? []);
  readonly notFound = computed(() => isNotFound(this.detail.error()));
  readonly error = computed(() => (this.detail.error() ? errorMessage(this.detail.error()) : null));

  reload(): void {
    this.detail.reload();
  }
}
